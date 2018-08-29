# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import numpy as np
import utils
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from BatchGeneration import Batch
FLAGS = None


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        # x_image = tf.reshape(x, [-1, 28, 28, 1])
        x_image = tf.reshape(x, [-1, FLAGS.width, FLAGS.height, FLAGS.channels])
        tf.summary.image("image", x_image)
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, FLAGS.channels, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([int(np.floor(FLAGS.width/4)*np.floor(FLAGS.height/4)*64), 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, int(np.floor(FLAGS.width/4)*np.floor(FLAGS.height/4)*64)])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # Map the 1024 features to  classes
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, FLAGS.nb_classes])
        b_fc2 = bias_variable([FLAGS.nb_classes])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def train_by_epoch(dataset_batch, model_eval, train_step, tf_data_input,
                   tf_true_label, keep_prob, nb_epochs=100, batch_size=50):
    def feed_dict(train=False, validation=False):
        """
        Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
        function extracted from https://www.tensorflow.org/get_started/summaries_and_tensorboard
        """
        if validation:
            xs, ys = dataset_batch.validation.images, dataset_batch.validation.labels
            k = 1.0
        elif train:
            xs, ys = dataset_batch.train.next_batch(batch_size)
            k = FLAGS.dropout
        else:
            xs, ys = dataset_batch.test.images, dataset_batch.test.labels
            k = 1.0
        return {tf_data_input: xs, tf_true_label: ys, keep_prob: k}

    sess = tf.InteractiveSession()
    merged_summary = tf.summary.merge_all()
    tf.global_variables_initializer().run()

    prev_epoch = dataset_batch.current_epoch
    iteration = 0
    train_writer = tf.summary.FileWriter('./tmp/train'+FLAGS.train_name, sess.graph)
    validation_writer = tf.summary.FileWriter('./tmp/validation'+FLAGS.train_name)
    test_writer = tf.summary.FileWriter('./tmp/test' + FLAGS.train_name)

    while dataset_batch.current_epoch < nb_epochs:
        if prev_epoch == dataset_batch.current_epoch:
            prev_epoch = dataset_batch.current_epoch + 1
            #evaluate on validation
            summary, acc = sess.run([merged_summary, model_eval], feed_dict=feed_dict(validation=True))
            print ("e:{} Val Acc.:{}".format(dataset_batch.current_epoch, acc))
            #train on validation
            sess.run([merged_summary, train_step], feed_dict=feed_dict(validation=True))
        else:  # Record train set summaries, and train
            iteration += 1
            if iteration % 50 == 0:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged_summary, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % iteration)
                train_writer.add_summary(summary, iteration)
                #evaluate on train
                _, acc = sess.run([merged_summary, model_eval], feed_dict=feed_dict(train=False))
                print ("\tepc:{} itr:{} Train Acc.:{}".format(dataset_batch.current_epoch, iteration, acc))
            else:  # Record a summary
                summary_train, _ = sess.run([merged_summary, train_step], feed_dict=feed_dict(train=True))
                _, acc_train = sess.run([merged_summary, model_eval], feed_dict=feed_dict(train=False))
                train_writer.add_summary(summary_train, iteration)

                # evaluate on validation
                summary_val, acc_val = sess.run([merged_summary, model_eval], feed_dict=feed_dict(validation=True))
                validation_writer.add_summary(summary_val, iteration)

                # evaluate on test
                summary_test, acc_test = sess.run([merged_summary, model_eval],
                                        feed_dict=feed_dict(train=False, validation=False))
                test_writer.add_summary(summary_test, iteration)

                print ("\tepc:{} itr:{} Train_acc:{} Val_acc:{} Test_acc:{}".format(dataset_batch.current_epoch, iteration, acc_train, acc_val, acc_train))

    summary, acc = sess.run([merged_summary, model_eval], feed_dict=feed_dict(train=False, validation=False))
    test_writer.add_summary(summary, prev_epoch)
    print ("e:{} Test acc: {}".format(dataset_batch.current_epoch, acc))


def main(_):
    # Create the model
    x = tf.placeholder(tf.float32, [None, FLAGS.width*FLAGS.height*FLAGS.channels], name='x-input')

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, FLAGS.nb_classes], name='y-input')

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    # Import data
    # dataset_batch = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    X, y = utils.read_data_file(FLAGS.data_dir)
    print ("file loaded")
    dataset_batch = Batch(X, y, test_perc=0.2, validation_perc=0.2)
    print ("Batch generated")
    print ("Test: ", dataset_batch.test.images.shape)
    print ("Validation: ", dataset_batch.validation.images.shape)
    print ("Train: ", dataset_batch.train.images.shape)

    # run_by_epoch(tf, dataset_batch=mnist)
    train_by_epoch(dataset_batch=dataset_batch,
                  model_eval=accuracy,
                  train_step=train_step,
                  tf_data_input=x,
                  tf_true_label=y_,
                  keep_prob=keep_prob,
                  nb_epochs=2,
                  batch_size=70)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for i in range(20000): #originally it was 20000
    #         batch = mnist.train.next_batch(50)
    #         if i % 10 == 0:
    #             train_accuracy = accuracy.eval(feed_dict={
    #                 x: batch[0], y_: batch[1], keep_prob: 1.0})
    #             print('step %d, training accuracy %g' % (i, train_accuracy))
    #         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #
    #     print('test accuracy %g' % accuracy.eval(feed_dict={
    #         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    is_real_mnist = False
    if is_real_mnist:
        parser.add_argument('--data_dir', type=str,
                            default='/tmp/tensorflow/mnist/input_data',
                            help='Directory for storing input data')
        parser.add_argument('--width', type=int,
                            default=28)
        parser.add_argument('--height', type=int,
                            default=28)
        parser.add_argument('--channels', type=int,
                            default=1)
        parser.add_argument('--nb_classes', type=int,
                            default=10)
        parser.add_argument('--dropout', type=float, default=0.9)
    else:
        parser.add_argument('--train_name', type=str,
                            default='data_100ptc_25size_isonehot_10epochs')
        parser.add_argument('--data_dir', type=str,
                            default='../data/data_100ptc_25size_isonehot.hdf5')
        parser.add_argument('--width', type=int,
                            default=144)
        parser.add_argument('--height', type=int,
                            default=180)
        parser.add_argument('--channels', type=int,
                            default=3)
        parser.add_argument('--nb_classes', type=int,
                            default=3)
        parser.add_argument('--dropout', type=float, default=0.9)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)#, argv=[sys.argv[0]] + unparsed)