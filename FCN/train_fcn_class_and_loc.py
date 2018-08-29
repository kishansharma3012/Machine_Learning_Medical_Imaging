import tensorflow as tf
from utils import normalize_tensor, get_max_xy

class TrainerAll(object):
    def __init__(self, dataset_batch, model_eval, model_eval_c, 
                 train_name, train_step, train_step_c, train_step_class_and_loc,
                 tf_data_input, tf_true_label, tf_true_label_c, sess=None, feed_dict=None,
                 batch_size=100, check_point_name=None, initialize_variables=True, args=None):
        self.sess = sess
        self.dataset_batch = dataset_batch
        self.model_eval = model_eval
        self.model_eval_c = model_eval_c
        self.train_name = train_name
        self.train_step = train_step
        self.train_step_c = train_step_c
        self.train_step_class_and_loc = train_step_class_and_loc
        self.tf_data_input = tf_data_input
        self.tf_true_label = tf_true_label
        self.tf_true_label_c = tf_true_label_c
        self.feed_dict = feed_dict
        self.batch_size = batch_size
        self.args = args
        self.check_point_name = check_point_name

        if not self.args:
            self.args = {}
        if not self.feed_dict:
            self.feed_dict = self.default_feed_dict
        if not self.sess:
            self.sess = tf.InteractiveSession()
        if initialize_variables:
            tf.global_variables_initializer().run()

        print(self.train_name)

    def default_feed_dict(self, train=False, validation=False, nexBatch=True, 
                          whole_data=False, classification = False, localization = False):
        """
        Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
        function extracted from https://www.tensorflow.org/get_started/summaries_and_tensorboard
        """
        
        if classification and localization:
            if validation:
                if whole_data:
                    xs, ys, ys_c = self.dataset_batch.validation.images, self.dataset_batch.validation.labels, self.dataset_batch.validation.labels_c
                else:
                    xs, ys, ys_c = self.dataset_batch.validation.next_batch(self.batch_size)
                k = 1.0
            elif train:
                if nexBatch:
                    xs, ys, ys_c = self.dataset_batch.train.next_batch(self.batch_size)
                else:
                    xs, ys, ys_c = self.dataset_batch.train.current_batch
                k = self.args.get('dropout', 0.5)
            else:
                if whole_data:
                    xs, ys, ys_c = self.dataset_batch.test.images, self.dataset_batch.test.labels, self.dataset_batch.test.labels_c
                else:
                    xs, ys, ys_c = self.dataset_batch.test.next_batch(self.batch_size)
                k = 1.0
            if self.args.get("keep_prob") is not None:
                return {self.tf_data_input: xs, self.tf_true_label: ys, self.tf_true_label_c: ys_c, self.args.get("keep_prob"): k}
            else:
                return {self.tf_data_input: xs, self.tf_true_label: ys, self.tf_true_label_c: ys_c}
        
    def apply_over_batches(self, writter, merged_summary, function,function_c,train=False, validation=False, 
                           classification = False, localization = False):
        batch = self.dataset_batch.test
        if validation:
            batch = self.dataset_batch.validation
        elif train:
            batch = self.dataset_batch.train
        prev_epoch = batch.current_epoch

        iteration = 0
        acc_avg = 0.0
        acc_avg_c = 0.0
        while prev_epoch == batch.current_epoch:
            iteration += 1
            summary, acc, acc_c = self.sess.run([merged_summary, function, function_c],
                                         feed_dict=self.feed_dict(train=train, validation=validation, whole_data=False,
                                                                  classification=classification, localization = localization))

            writter.add_summary(summary, iteration)
            acc_avg   += acc 
            acc_avg_c += acc_c
            print ("Testing {}/{}: localization {:.3} classification {:.3}".format(iteration, int(batch.num_examples/self.batch_size), acc, acc_c))
        writter.flush()
        if iteration > 0:
            acc_avg /= float(iteration)
            acc_avg_c /= float(iteration)

            summary1 = tf.Summary()
            summary2 = tf.Summary()

            summary1.value.add(tag='avg_accuracy_loc', simple_value=acc_avg)
            summary2.value.add(tag='avg_accuracy_class', simple_value=acc_avg_c)

            writter.add_summary(summary1, iteration + 1)
            writter.add_summary(summary2, iteration + 1)
            writter.flush()
        else:
            print ("There was an erorr with testing. No interation happened while testing.")

    def train_by_epoch(self, nb_epochs, save_checkpoint=None, classification = False, localization = False):
        merged_summary = tf.summary.merge_all()

        checkpointSaver = None
        if save_checkpoint:
            checkpointSaver = tf.train.Saver()

        iteration = 0
        train_writer = tf.summary.FileWriter('./21_01_classif/train' + self.train_name, self.sess.graph)
        validation_writer = tf.summary.FileWriter('./21_01_classif/validation' + self.train_name)
        test_writer = tf.summary.FileWriter('./21_01_classif/test' + self.train_name)

        prev_epoch = 0
        while self.dataset_batch.current_epoch < nb_epochs:
            if prev_epoch < self.dataset_batch.current_epoch:
                if save_checkpoint and checkpointSaver:
                    prev_epoch += 1
                    saved_path = self.save_check_point(checkpointSaver)
                    print ("Model {} in epoch {} was saved at: {}".format(
                                                                    self.train_name,
                                                                    self.dataset_batch.current_epoch,
                                                                     saved_path)
                                                                    )
            iteration += 1
            
            summary_train, _ , _ , _= self.sess.run([merged_summary, self.train_step, self.train_step_c, self.train_step_class_and_loc],
                                                                    feed_dict=self.feed_dict(train=True,classification=classification,
                                                                                             localization=localization))
            
            train_writer.add_summary(summary_train, iteration)

            train_writer.flush()
            
            _, acc_train, acc_train_c = self.sess.run([merged_summary, self.model_eval, self.model_eval_c],
                                         feed_dict=self.feed_dict(train=True, nexBatch=False,classification=classification,
                                                                  localization=localization))
            
            summary_val, acc_val, acc_val_c= self.sess.run([merged_summary, self.model_eval, self.model_eval_c],
                                                 feed_dict=self.feed_dict(validation=True,classification=classification,
                                                                          localization=localization))
        
            validation_writer.add_summary(summary_val, iteration)
            validation_writer.flush()
            
            print ("\tepc:{} itr:{} Train_acc_loc:{:.3} Val_acc_loc:{:.3} Train_acc_c:{:.3} Val_acc_c:{:.3} ".format(self.dataset_batch.current_epoch,
                                                                    iteration,
                                                                    acc_train,
                                                                    acc_val,
                                                                    acc_train_c,
                                                                    acc_val_c))
        if save_checkpoint:
            saved_path = self.save_check_point(checkpointSaver)
            print ("Model {} in epoch {} was saved at: {}".format(
                                                self.train_name,
                                                self.dataset_batch.current_epoch,
                                                saved_path)
                                            )
        self.apply_over_batches(test_writer, merged_summary, self.model_eval, self.model_eval_c, train=False, validation=False, 
                                classification=classification, localization=localization)


    def restore_from_check_point(self, checkpoint_path=None):
        checkpointSaver = tf.train.Saver()
        if not checkpoint_path:
            checkpoint_path = "./checkpoints/{}/checkpoint.ckpt".format(self.train_name)
        checkpointSaver.restore(self.sess, checkpoint_path)
        print ("Checkpoint restored {}".format(checkpoint_path))
        return self

    def save_check_point(self, checkpointSaver, checkpoint_path=None):
        if checkpoint_path:
            return checkpointSaver.save(self.sess, checkpoint_path, global_step=self.dataset_batch.current_epoch)
        return checkpointSaver.save(self.sess, "./checkpoints/{}/checkpoint".format(self.train_name),
                                                                              global_step=self.dataset_batch.current_epoch)


    @classmethod
    def loss_function(self, prediction, labels, cost_name="l2", transpose=False, args=None):
        if transpose:
            labels = tf.transpose(labels, [0, 3, 1, 2]) # change dims to N,12,224,224
            prediction = tf.transpose(prediction, [0, 3, 1, 2]) # change dims to N,12,224,224

        flat_logits_true = tf.reshape(prediction, [-1, 12, args.width*args.height])
        flat_logits = tf.reshape(labels, [-1, 12, args.width*args.height])


        if cost_name == "softmax_cross_entropy":
            # Now for each sample, we have 12 joints, and for each joint 224*224 classes
            loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits_true, labels=flat_logits), 1))
        elif cost_name == "l2":
            diff = normalize_tensor(flat_logits_true, 1) - normalize_tensor(flat_logits, 1)
            loss = tf.nn.l2_loss(diff)
            #loss = tf.reduce_mean(tf.pow(diff, 2))
            
        else:
            raise ValueError("Unknown cost function: " % cost_name)
        return loss
   
    @classmethod
    def accuracy_function(self, prediction, labels, acc_name='locali', threshold = 15, transpose=False, args=None, verbose=False):
   
        """
        Expects: [-1, W, H, 12] if transpose=True
                 [-1, 12, W, H] if transpose=False
        """
        if transpose:
            labels = tf.transpose(labels, [0, 3, 1, 2])  # change dims to N,12,224,224
            prediction = tf.transpose(prediction, [0, 3, 1, 2])  # change dims to N,12,224,224
    
        if acc_name == 'argmax':
            x_gt, y_gt = get_max_xy(labels, k=1)
            x_pr, y_pr = get_max_xy(prediction, k=1)
    
            if verbose:
                x_pr = tf.Print(x_pr, [x_pr[0][0], y_pr[0][0], x_pr[0][1], y_pr[0][1]], "x_pr, y_pr: ")
                x_gt = tf.Print(x_gt, [x_gt[0][0], y_gt[0][0], x_gt[0][1], y_gt[0][1]], "x_gt, y_gt: ")
    
            correct_prediction_x = tf.sqrt(tf.reduce_mean(tf.squared_difference(x_gt, x_pr)))
            correct_prediction_y = tf.sqrt(tf.reduce_mean(tf.squared_difference(y_gt, y_pr)))
            correct_prediction = 1. - tf.divide((correct_prediction_x + correct_prediction_y),
                                                (args.width + args.height))
    
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        elif acc_name == 'locali':
            if isinstance(args, dict):
                threshold = args.get("threshold", threshold)
            threshold = tf.cast(tf.constant(threshold), tf.float32)
            x, y = get_max_xy(prediction, k=1, dtype=tf.float32) #N*12
            x_l, y_l = get_max_xy(labels, k=1, dtype=tf.float32) #N*12
            x = tf.reshape(x, [-1, 12])
            y = tf.reshape(y, [-1, 12])
            x_l = tf.reshape(y_l, [-1, 12])
            y_l = tf.reshape(y_l, [-1, 12])

            final_x = tf.square(tf.subtract(x, x_l))
            final_y = tf.square(tf.subtract(y, y_l))

            final = tf.sqrt(tf.add(final_x, final_y))
            f = tf.less(final, threshold)
            correct_prediction =  tf.reduce_mean(tf.cast(f, 'float'))
        else:
            raise ValueError("Unknown acc function: " % acc_name)
        return correct_prediction
