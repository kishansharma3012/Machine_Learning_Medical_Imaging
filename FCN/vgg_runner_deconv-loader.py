import vgg
from BatchGeneration import Batch
import argparse
import tempfile
import tensorflow as tf
from train_fcn_loader import Trainer

# from train_fcn_class_and_loc import TrainerAll


slim = tf.contrib.slim
from time import gmtime, strftime

from utils import normalize_tensor, add_imgs_to_summary, produce_sum_maps, get_images_bb_max_xy


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # Create the model BOTH CLASSIFICATION AND LOCALIZATION
    x = tf.placeholder(tf.float32, [None, FLAGS.width, FLAGS.height, FLAGS.channels], name='x-input')
    tf.summary.image("input", x, max_outputs=1)  # , family="ground_truth")

    

    # LOCALIZATION
    y_gaussian = tf.placeholder(tf.float32, [None, FLAGS.nb_maps, FLAGS.width, FLAGS.height], name='y-input')

    #y_gaussian = tf.transpose(y_gaussian, [0, 2, 3, 1])
    #y_gaussian = tf.reshape(y_gaussian, [-1, FLAGS.width, FLAGS.height, 12])  # N, Width, height, Ins.*Joints
    #y_gaussian = tf.transpose(y_gaussian, [0, 3, 1, 2])  # N, Ins.*Joints, Width, height
    y_gaussian_reshape = tf.expand_dims(y_gaussian, -1)  # N, Ins.*Joints, Width, height, 1

    y_gaussian_prediction_img_sum = tf.reduce_sum(y_gaussian_reshape, 1)
    tf.summary.image("gt_img_sum", y_gaussian_prediction_img_sum, max_outputs=1)

    add_imgs_to_summary(y_gaussian_reshape, range(FLAGS.nb_maps), base_name="gt/img")

    # Build the network
    y_conv = vgg.vgg_only_conv(x, verbose=True)
    y_conv = vgg.vgg_only_deconv(y_conv, verbose=True, args=FLAGS)  # outputs: [-1, 224, 224, 6, 2]
    y_convt = tf.reshape(y_conv, [-1, FLAGS.width, FLAGS.height, FLAGS.nb_maps])
    y_convt = tf.transpose(y_convt, [0, 3, 1, 2])


    y_conv_normalized = normalize_tensor(y_conv)
    y_conv_sum, y_conv_reshape = produce_sum_maps(y_conv_normalized, FLAGS)
    prediction_bb = get_images_bb_max_xy(y_conv_reshape, gaussian_idx_list=range(FLAGS.nb_maps), k=1, args=FLAGS)
    add_imgs_to_summary(prediction_bb, range(FLAGS.nb_maps), base_name="prediction/bb")

    print ("Threshold\tIdx Joint\tAcc.")
    for idx in range(FLAGS.nb_maps):
        for t in [5,10,15,20,25,30]:
            # Loss and accuracy
                with tf.name_scope('accuracy'):
                    with tf.name_scope("prediction"):
                        correct_prediction = Trainer.accuracy_function(labels=y_gaussian,
                                                                       prediction=y_convt,
                                                                       acc_name='locali2',
                                                                       threshold=t,
                                                                       joint = idx,
                                                                       transpose=False,
                                                                       verbose=False,
                                                                       args=FLAGS)

                    with tf.name_scope("accuracy"):
                        accuracy = tf.reduce_mean(correct_prediction)
                    accuracy = tf.Print (accuracy, ["{} {}".format(t, idx), accuracy], "")
                    tf.summary.scalar('accuracy_localization_{}_{}'.format(t, idx), accuracy)

    # Loss and accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope("prediction"):
            correct_prediction = Trainer.accuracy_function(labels=y_gaussian,
                                                           prediction=y_convt,
                                                           acc_name='locali',
                                                           threshold=15,
                                                           transpose=False,
                                                           verbose=False,
                                                           args=FLAGS)

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(correct_prediction)


        tf.summary.scalar('accuracy_localization_15', accuracy)



    with tf.name_scope("loss"):
        loss = Trainer.loss_function(labels=y_gaussian,
                                     prediction=y_convt,
                                     cost_name="softmax_cross_entropy",
                                     transpose=False,
                                     args=FLAGS)
    tf.summary.scalar('loss_localization', loss)

    with tf.name_scope('adam_optimizer'):
        #train_step_loc = tf.train.AdamOptimizer(1e-2).minimize(loss)
        optimizer = tf.train.AdamOptimizer(1e-5)
        grads = tf.gradients(loss, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
        # Op to update all variables according to their gradient
        train_step_loc = optimizer.apply_gradients(grads_and_vars=grads)
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)


    # INITIALIZATION AND BATCH GENERATION
    # Create the initialization function for the variables of the network
    init_op = tf.global_variables_initializer()
    '''variables_to_restore = slim.get_variables_to_restore(
        exclude=['vgg_16/fc[\d]+', 'vgg_16/deconv1/*', '.*adam_optimizer.*', 'vgg_16/conv1/*',
                 '.*[Aa]dam.*', ""
                                'vgg_16/(other_|original_)?up[\d].*','vgg_16/(other_|original_)?conv_up[\d].*',
                 "a_____out_gauss", "save/RestoreV2/.*", "variable.*"])
    '''
    # Define an operator to load model variables from a checkpoint using Slim.
    # The checkpoint can be found at https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
    #checkpoint_path = './checkpoints/_10joints_100pct_20ep'
    # checkpoint_path = '../data/vgg_16.ckpt'
    #restorer = tf.train.Saver(slim.get_variables_to_restore())

    print("init...")
    sess.run(init_op)
    print("restoring...")
    #restorer.restore(sess, checkpoint_path)
    print("restroring done")

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    dataset_batch = Batch(file_path=FLAGS.data_dir_train, validation_perc=0.05,
                          test_perc=0.5)  # , test_perc=0.9)  # , test_perc=0.60)
    dataset_batch_test = Batch(file_path=FLAGS.data_dir_test, test_perc=0.99, validation_perc=0.01)

    dataset_batch.test = dataset_batch_test.test
    # dataset_batch.validation = dataset_batch_test.validation

    print("Batch generated")

    print("Test: ", dataset_batch.test.images.shape, dataset_batch.test.min_idx, dataset_batch.test.max_idx)
    print("Validation: ", dataset_batch.validation.images.shape, dataset_batch.validation.min_idx,
          dataset_batch.validation.max_idx)
    print("Train: ", dataset_batch.train.images.shape, dataset_batch.train.min_idx, dataset_batch.train.max_idx)

    # TRAINER CLASSIFICATION OR LOCALIZATION
    '''
    #For classification:
    Trainer(dataset_batch=dataset_batch,
            model_eval=accuracy_class, train_step=train_step_class, tf_data_input=x,
            tf_true_label=y_class, batch_size=20,
            sess=sess, train_name=FLAGS.train_name, initialize_variables=False
            ).train_by_epoch(10, save_checkpoint=False, classification=True)

    '''
    # '''
    # For localization:
    Trainer(dataset_batch=dataset_batch,
            model_eval=accuracy, train_step=train_step_loc, tf_data_input=x,
            tf_true_label=y_gaussian, batch_size=20,
            sess=sess, train_name=FLAGS.train_name, initialize_variables=False
            ).restore_from_check_point().train_by_epoch(FLAGS.epochs, save_checkpoint=False, classification=False)
    # '''

    '''
    # TRAINER CLASSIFICATION AND LOCALIZATION
    TrainerAll(dataset_batch=dataset_batch,
            model_eval=accuracy, model_eval_c = accuracy_class,
            train_step=train_step_loc, train_step_c = train_step_class, train_step_class_and_loc=train_step_class_and_loc,
            tf_data_input=x, tf_true_label=y_gaussian, tf_true_label_c = y_class,
            batch_size=20, sess=sess, train_name=FLAGS.train_name, initialize_variables=False
            ).train_by_epoch(5, save_checkpoint=False, classification=True, localization=True)

     '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_name', type=str,
                        default='_10joints_getting_points' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    parser.add_argument('--data_dir_train', type=str,
                        default='./data/train_class+loc_images_224_10joints_chuncked.hdf5')
    parser.add_argument('--data_dir_test', type=str,
                        default='./data/test_class+loc_images_224_10joints_chuncked.hdf5')
    parser.add_argument('--width', type=int,
                        default=224)
    parser.add_argument('--height', type=int,
                        default=224)
    parser.add_argument('--channels', type=int,
                        default=3)
    parser.add_argument('--nb_classes', type=int,
                        default=4)
    parser.add_argument('--nb_maps', type=int,
                        default=10)
    parser.add_argument('--dropout', type=float, default=0.9)

    parser.add_argument('--lambda_black', type=float, default=0.2)
    parser.add_argument('--lambda_acc', type=float, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)  # , argv=[sys.argv[0]] + unparsed)
