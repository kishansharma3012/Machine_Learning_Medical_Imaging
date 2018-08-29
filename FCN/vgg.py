# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import functools

slim = tf.contrib.slim


def normalize_tensor(tensor_val):
    return tf.multiply(tf.div(
        tf.subtract(
            tensor_val,
            tf.reduce_min(tensor_val)
        ),
        tf.add(tf.subtract(
            tf.reduce_max(tensor_val),
            tf.reduce_min(tensor_val)
        ), 0.0001)
    ), 255.)


def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

def get_bilinear_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights

def upsample_layer(bottom,
                       n_channels, num_out_channels, name, upscale_factor):
        kernel_size = 2 * upscale_factor - upscale_factor % 2
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            # n_channels = num_out_channels
            # Shape of the bottom tensor
            in_shape = tf.shape(bottom)

            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = in_shape[1] * stride
            w = in_shape[2] * stride

            new_shape = [in_shape[0], h, w, n_channels]
            output_shape = tf.stack(new_shape)

            filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

            weights = get_bilinear_filter(filter_shape, upscale_factor)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')
            deconv = slim.conv2d(deconv, num_out_channels, [1, 1], padding="SAME", scope=name + "_conv")


        return deconv

#REFERENCE: https://github.com/tensorflow/models/blob/master/research/slim/nets/pix2pix.py
def upsample(net, num_outputs, kernel_size, scope="", method='nn_upsample_conv'):
    """Upsamples the given inputs.
    Args:
      net: A `Tensor` of size [batch_size, height, width, filters].
      num_outputs: The number of output filters.
      kernel_size: A list of 2 scalars or a 1x2 `Tensor` indicating the scale,
        relative to the inputs, of the output dimensions. For example, if kernel
        size is [2, 3], then the output height and width will be twice and three
        times the input size.
      method: The upsampling method.
    Returns:
      An `Tensor` which was upsampled using the specified method.
    Raises:
      ValueError: if `method` is not recognized.
    """
    net_shape = tf.shape(net)
    height = net_shape[1]
    width = net_shape[2]
    with tf.name_scope(scope):
        if method == 'nn_upsample_conv':
            net = tf.image.resize_nearest_neighbor(net, [kernel_size[0] * height, kernel_size[1] * width])
            net = tf.contrib.layers.conv2d(net, num_outputs, [4, 4], activation_fn=tf.nn.relu, scope=scope+"_conv")
        elif method == 'conv2d_transpose':
            net = tf.contrib.layers.conv2d_transpose(net, num_outputs, [4, 4], stride=kernel_size, activation_fn=tf.nn.relu)
        elif method == 'bilinear_upsample_conv':
            net = tf.image.resize_bilinear(net, [kernel_size[0] * height, kernel_size[1] * width])
            net = tf.contrib.layers.conv2d(net, num_outputs, [4, 4], activation_fn=tf.nn.relu, scope=scope + "_conv")
        else:
            raise ValueError('Unknown method: [%s]', method)

    return net

def vgg_only_conv(inputs, scope='vgg_16',
               fc_conv_padding='SAME',
               dropout_keep_prob=0.5,
               is_training=True,
               verbose=False,
               classif=False):
    """
    Args:
        inputs: a tensor of size [batch_size, width, height, channels]
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.

    Returns:
        net: tensor in shape [batch_size, width, height, number_of_joints]
    """

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        ########################################################
                        ## ENCODING ##
        ########################################################
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', variables_collections="conv1_collection")
            if verbose: print(net.shape, 'conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            if verbose: print(net.shape, 'pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            if verbose: print(net.shape, 'conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            if verbose: print(net.shape, 'pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            if verbose: print(net.shape, 'conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            if verbose: print(net.shape, 'pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            if verbose: print(net.shape, 'conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            if verbose: print(net.shape, 'pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if verbose: print(net.shape, 'conv5')
            
            if classif==True:
                net = slim.max_pool2d(net, [2, 2], scope='pool5') # it turns to 7x7
                if verbose: print(net.shape, 'pool5')
            
            net = slim.conv2d(net, 1024, [7, 7], padding=fc_conv_padding, scope='fc6')
            if verbose: print(net.shape, 'fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

            net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
            if verbose: print(net.shape, 'fc7')


            #TODO: experiment 2 - > comment this one 14x14
            # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            # net = slim.conv2d(net, 4, [1, 1], activation_fn=tf.nn.relu, normalizer_fn=None, scope='fc8')
            # if verbose: print(net.shape, 'fc8')

            net_norm = normalize_tensor(net)
            net_norm = tf.reshape(net_norm, [-1, 7, 7, 4, 1])
            tf.summary.image("features_fc8_0", net_norm[:, :, :, 0, :], max_outputs=1)#, family="features")
            tf.summary.image("features_fc8_1", net_norm[:, :, :, 1, :], max_outputs=1)#, family="features")
            tf.summary.image("features_fc8_2", net_norm[:, :, :, 2, :], max_outputs=1)#, family="features")
            tf.summary.image("features_fc8_3", net_norm[:, :, :, 3, :], max_outputs=1)#, family="features")
            
        return net
    
    
def vgg_only_deconv(inputs, scope='vgg_16',
               fc_conv_padding='SAME',
               dropout_keep_prob=0.5,
               is_training=True,
               verbose=False,
               args=None):
    """
    Args:
        inputs: a tensor of size [batch_size, width, height, channels]
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.

    Returns:
        net: tensor in shape [batch_size, width, height, number_of_joints]
    """
    upsample_fn = functools.partial(upsample, method='bilinear_upsample_conv')

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
           
            ##############
            ## DECODING ##
            ##############
            net = slim.conv2d(inputs, 1024, [1, 1], activation_fn=tf.nn.relu, normalizer_fn=None, scope='deconv1')
            if verbose: print(net.shape, 'deconv1')

            # net = upsample_fn(net, 256, [2,2], scope="other_up1")
            # net = tf.reshape(net, [tf.shape(net)[0], 14, 14, 256])

            net_norm = normalize_tensor(net)
            net_norm = tf.reshape(net_norm, [-1, 14, 14, 1024, 1])
            tf.summary.image("features_up1_0", net_norm[:, :, :, 0, :], max_outputs=1 )# family="features")
            tf.summary.image("features_up1_1", net_norm[:, :, :, 16, :], max_outputs=1) #, family="features")
            tf.summary.image("features_up1_2", net_norm[:, :, :, 32, :], max_outputs=1)# family="features")
            tf.summary.image("features_up1_3", net_norm[:, :, :, 255, :], max_outputs=1)#, family="features")

            if verbose: print(net.shape, 'up1')
            net = upsample_fn(net, 512, [2, 2], scope="up2")
            net = slim.repeat(net, 2, slim.conv2d, 512, [7, 7], scope='conv_up1')
            net = tf.reshape(net, [tf.shape(net)[0], 28, 28, 512])
            if verbose: print(net.shape, 'up2')
            net = upsample_fn(net, 128, [2, 2], scope="up3")
            net = slim.repeat(net, 2, slim.conv2d, 128, [7, 7], scope='conv_up2')
            net = tf.reshape(net, [tf.shape(net)[0], 56, 56, 128])
            if verbose: print(net.shape, 'up3')
            net = upsample_fn(net, 32, [2, 2], scope="up4")
            net = slim.repeat(net, 2, slim.conv2d, 32, [7, 7], scope='conv_up3')
            net = tf.reshape(net, [tf.shape(net)[0], 112, 112, 32])

            net_norm = normalize_tensor(net)
            net_norm = tf.reshape(net_norm, [-1, 112, 112, 32, 1])
            tf.summary.image("features_up4_0", net_norm[:, :, :, 0, :], max_outputs=1 )#, family="features")
            tf.summary.image("features_up4_1", net_norm[:, :, :, 3, :], max_outputs=1 )#, family="features")
            tf.summary.image("features_up4_2", net_norm[:, :, :, 7, :], max_outputs=1)#, family="features")
            tf.summary.image("features_up4_3", net_norm[:, :, :, 11, :], max_outputs=1)#, family="features")

            if verbose: print(net.shape, 'up4')
            net = upsample_fn(net, args.nb_maps, [2, 2], scope="up5")
            if args.nb_maps == 10:
                net = tf.reshape(net, [tf.shape(net)[0], 224, 224, 5, 2])
            elif args.nb_maps == 12:
                net = tf.reshape(net, [tf.shape(net)[0], 224, 224, 6, 2])
            elif args.nb_maps == 20:
                net = tf.reshape(net, [tf.shape(net)[0], 224, 224, 5, 4])
            else:
                print("sth. wrong with the number of output maps...")
            if verbose: print(net.shape, 'up5')

        return net
def vgg_finish_conv(inputs,
                    num_classes=1000,
                    is_training=True,
                    dropout_keep_prob=0.5,
                    spatial_squeeze=True,
                    scope='vgg_16',
                    fc_conv_padding='VALID',
                    verbose=False):
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
          
          if num_classes:
            net = slim.dropout(inputs, dropout_keep_prob, is_training=is_training,
                               scope='dropout8')
            
            net = slim.conv2d(net, num_classes, [7, 7],
                              padding=fc_conv_padding,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=None,
                              scope='fc9')
            if verbose: print(net.shape, 'fc9')
            if spatial_squeeze and num_classes is not None:
              net = tf.squeeze(net, [1, 2], name='fc9/squeezed')
            if verbose: print(net.shape, 'end')
        return net
    


def vgg_deconv(inputs, scope='vgg_16',
               fc_conv_padding='SAME',
               dropout_keep_prob=0.5,
               is_training=True,
               verbose=False):
    """
    Args:
        inputs: a tensor of size [batch_size, width, height, channels]
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.

    Returns:
        net: tensor in shape [batch_size, width, height, number_of_joints]
    """
    upsample_fn = functools.partial(upsample, method='bilinear_upsample_conv')

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        ########################################################
                        ## ENCODING ##
        ########################################################
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', variables_collections="conv1_collection")
            if verbose: print(net.shape, 'conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            if verbose: print(net.shape, 'pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            if verbose: print(net.shape, 'conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            if verbose: print(net.shape, 'pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            if verbose: print(net.shape, 'conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            if verbose: print(net.shape, 'pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            if verbose: print(net.shape, 'conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            if verbose: print(net.shape, 'pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if verbose: print(net.shape, 'conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            if verbose: print(net.shape, 'pool5')
            
            # reduced from 4096 to 512
            net = slim.conv2d(net, 512, [7, 7], padding=fc_conv_padding, scope='fc6')
            if verbose: print(net.shape, 'fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 512, [1, 1], scope='fc7')
            if verbose: print(net.shape, 'fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = slim.conv2d(net, 4, [1, 1], activation_fn=tf.nn.relu, normalizer_fn=None, scope='fc8')
            if verbose: print(net.shape, 'fc8')

            net_norm = normalize_tensor(net)
            net_norm = tf.reshape(net_norm, [-1, 7, 7, 4, 1])
            tf.summary.image("features_fc8_0", net_norm[:, :, :, 0, :], max_outputs=1)#, family="features")
            tf.summary.image("features_fc8_1", net_norm[:, :, :, 1, :], max_outputs=1)#, family="features")
            tf.summary.image("features_fc8_2", net_norm[:, :, :, 2, :], max_outputs=1)#, family="features")
            tf.summary.image("features_fc8_3", net_norm[:, :, :, 3, :], max_outputs=1)#, family="features")
            

            ##############
            ## DECODING ##
            ##############
            
            net = slim.conv2d(net, 512, [1, 1], activation_fn=tf.nn.relu, normalizer_fn=None, scope='deconv1')
            if verbose: print(net.shape, 'deconv1')
            
            # net = upsample_layer(net, 512, 256, 'up1', 2)
            net = upsample_fn(net, 256, [2,2], scope="other_up1")
            net = tf.reshape(net, [tf.shape(net)[0], 14, 14, 256])

            net_norm = normalize_tensor(net)
            net_norm = tf.reshape(net_norm, [-1, 14, 14, 256, 1])
            tf.summary.image("features_up1_0", net_norm[:, :, :, 0, :], max_outputs=1 )# family="features")
            tf.summary.image("features_up1_1", net_norm[:, :, :, 16, :], max_outputs=1) #, family="features")
            tf.summary.image("features_up1_2", net_norm[:, :, :, 32, :], max_outputs=1)# family="features")
            tf.summary.image("features_up1_3", net_norm[:, :, :, 255, :], max_outputs=1)#, family="features")

            if verbose: print(net.shape, 'up1')
            # net = upsample_layer(net, 256, 128, 'up2', 2)
            net = upsample_fn(net, 128, [2, 2], scope="up2")
            net = tf.reshape(net, [tf.shape(net)[0], 28, 28, 128])
            if verbose: print(net.shape, 'up2')
            net = upsample_fn(net, 64, [2, 2], scope="up3")
            # net = upsample_layer(net, 128, 64, 'up3', 2)
            net = tf.reshape(net, [tf.shape(net)[0], 56, 56, 64])
            if verbose: print(net.shape, 'up3')
            net = upsample_fn(net, 32, [2, 2], scope="up4")
            # net = upsample_layer(net, 64, 32, 'up4', 2)
            net = tf.reshape(net, [tf.shape(net)[0], 112, 112, 32])

            net_norm = normalize_tensor(net)
            net_norm = tf.reshape(net_norm, [-1, 112, 112, 32, 1])
            tf.summary.image("features_up4_0", net_norm[:, :, :, 0, :], max_outputs=1 )#, family="features")
            tf.summary.image("features_up4_1", net_norm[:, :, :, 3, :], max_outputs=1 )#, family="features")
            tf.summary.image("features_up4_2", net_norm[:, :, :, 7, :], max_outputs=1)#, family="features")
            tf.summary.image("features_up4_3", net_norm[:, :, :, 11, :], max_outputs=1)#, family="features")

            if verbose: print(net.shape, 'up4')
            #new maps size: 10
            net = upsample_fn(net, 10, [2, 2], scope="up5")
            # net = upsample_layer(net, 32, 12, 'up5', 2)
            net = tf.reshape(net, [tf.shape(net)[0], 224, 224, 5, 2])
            if verbose: print(net.shape, 'up5')

        return net


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
    """Oxford Net VGG 11-Layers version A Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer is
        omitted and the input features to the logits layer are returned instead.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      fc_conv_padding: the type of padding to use for the fully connected layer
        that is implemented as a convolutional layer. Use 'SAME' padding if you
        are applying the network in a fully convolutional manner and want to
        get a prediction map downsampled by a factor of 32 as an output.
        Otherwise, the output prediction map will be (input / 32) - 6 in case of
        'VALID' padding.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original VGG architecture.)

    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the input to the logits layer (if num_classes is 0 or None).
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=None,
                                  scope='fc8')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=tf.nn.relu,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze and num_classes is not None:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
    """Oxford Net VGG 19-Layers version E Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer is
        omitted and the input features to the logits layer are returned instead.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      fc_conv_padding: the type of padding to use for the fully connected layer
        that is implemented as a convolutional layer. Use 'SAME' padding if you
        are applying the network in a fully convolutional manner and want to
        get a prediction map downsampled by a factor of 32 as an output.
        Otherwise, the output prediction map will be (input / 32) - 6 in case of
        'VALID' padding.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original VGG architecture.)

    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the non-dropped-out input to the logits layer (if num_classes is 0 or
        None).
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=None,
                                  scope='fc8')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19