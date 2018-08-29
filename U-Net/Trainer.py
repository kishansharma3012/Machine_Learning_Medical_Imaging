from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    """
    
    verification_batch_size = 4
    
    def __init__(self, net, batch_size=1, norm_grads=False, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.locali_cost, 
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 1e-04)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.locali_cost,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)

        loss_summary = tf.summary.scalar('loss', self.net.locali_cost)
        #accuracy_summary = tf.summary.scalar('accuracy', self.net.accuracy)
        locali_rmse = tf.summary.scalar('RMSE', self.net.locali_error)
        self.optimizer = self._get_optimizer(training_iters, global_step)
        learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()

        output_path = os.path.abspath(output_path)
        print(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
            logging.info("Allocating '{:}'".format(output_path + "/TrainSummary"))
            os.makedirs(output_path + "/TrainSummary")
            logging.info("Allocating '{:}'".format(output_path + "/ValSummary"))
            os.makedirs(output_path + "/ValSummary")
        
        return init, init2

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False, write_graph=False, prediction_path = 'prediction'):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init, init2 = self._initialize(training_iters, output_path, restore, prediction_path)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            sess.run(init2)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            train_summary_writer = tf.summary.FileWriter(output_path + "/TrainSummary", graph=sess.graph)
            val_summary_writer = tf.summary.FileWriter(output_path + "/ValSummary", graph=sess.graph)
            logging.info("Start optimization")
            for epoch in range(epochs):
                total_loss = 0
                total_pred10 = 0
                total_pred15 = 0
                total_pred20 = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_classi_y, batch_locali_y = data_provider.get_training_batch(self.batch_size)
                    print(batch_locali_y.shape)
                    print(batch_x.shape)
                    print(batch_classi_y.shape)
                    # Run optimization op (backprop)
                    _, loss, lr = sess.run((self.optimizer, self.net.locali_cost, self.learning_rate_node), 
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.locali_y: batch_locali_y,
                                                                 self.net.keep_prob: dropout})

                    pred10 = 0
                    pred15 = 0
                    pred20 = 0
                    if step % display_step == 0:
                        pred10, pred15, pred20 = self.output_minibatch_stats(sess, train_summary_writer, step, batch_x, batch_locali_y)
                        
                    total_loss += loss
                    total_pred10 += pred10
                    total_pred15 += pred15
                    total_pred20 += pred20

                    del batch_x
                    del batch_classi_y
                    del batch_locali_y

                self.output_epoch_stats(epoch, total_pred10, total_pred15, total_pred20, total_loss, training_iters, lr)
                self.output_validation_epoch_stats(data_provider, sess, val_summary_writer, epoch)
                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")
            
            return save_path

    def output_validation_epoch_stats(self, data_provider, sess, val_summary_writer, epoch):
        batchsize=10
        batches = data_provider.no_validation_batches(10)
        print("batches are:" + str(batches))
        total_loss = 0
        total_acc = 0
        total_error = 0
        for i in range(batches):
            val_x, val_classi_y, val_locali_y = data_provider.get_validation_batch(batchsize)
            loss, error, predictions = sess.run([self.net.locali_cost, 
                                              self.net.locali_error,
                                              self.net.locali_predicter], 
                                              feed_dict={self.net.x: val_x,
                                                         self.net.locali_y: val_locali_y,
                                                         self.net.keep_prob: 1.})

            pred10 = sess.run([self.net.locali_accuracy], feed_dict={self.net.x: val_x,
                                                                    self.net.locali_y: val_locali_y,
                                                                    self.net.threshold:tf.constant(10),
                                                                    self.net.keep_prob: 1.})

            pred15 = sess.run([self.net.locali_accuracy], feed_dict={self.net.x: val_x,
                                                                    self.net.locali_y: val_locali_y,
                                                                    self.net.threshold:tf.constant(15),
                                                                    self.net.keep_prob: 1.})

            pred20 = sess.run([self.net.locali_accuracy], feed_dict={self.net.x: val_x,
                                                                    self.net.locali_y: val_locali_y,
                                                                    self.net.threshold:tf.constant(20),
                                                                    self.net.keep_prob: 1.})
            
            total_loss += loss
            #total_acc += acc
            total_error += error #error_rate(predictions, val_classi_y)
            del val_x
            del val_classi_y
            del val_locali_y
        total_loss /= float(batches)
        #total_acc /= float(batches)
        total_error /= float(batches)
        
        # Write loss summary
        summary = tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=total_loss),])
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()

        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy10", simple_value=pred10),])
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()
        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy15", simple_value=pred15),])
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()
        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy20", simple_value=pred20),])
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()
        
        # Write accuracy summary
        #summary = tf.Summary(value=[tf.Summary.Value(tag="val_accuracy", simple_value=total_acc),])
        #val_summary_writer.add_summary(summary, epoch)
        #val_summary_writer.flush()
        
        # Write error summary
        summary = tf.Summary(value=[tf.Summary.Value(tag="RMSE", simple_value=total_error),])
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()


        #Precision= {:.4f}, Recall= {:.4f}, 
        logging.info("VALIDATION: epoch {:}, validation Loss= {:.8f}, accuracy10= {:.8f}, accuracy15= {:.8f}, accuracy20= {:.8f}, validation RMSE error= {:.8f}%".format(epoch,
                                                                                        total_loss,
                                                                                        pred10,
                                                                                        pred15,
                                                                                        pred20,
                                                                                        total_error))
            
    def output_epoch_stats(self, epoch, total_pred10, total_pred15, total_pred20, total_loss, training_iters, lr):
        logging.info("Training: Epoch {:}, Average loss: {:.8f}, Average pred10{:.8f}, Average pred15{:.8f}, Average pred20{:.8f}, \
                      learning rate: {:.8f}".format(epoch, (total_loss / training_iters), (total_pred10 / training_iters),
                      (total_pred15 / training_iters), (total_pred20 / training_iters), lr))
    
    def output_minibatch_stats(self, sess, train_summary_writer, step, batch_x, batch_locali_y):
        # Calculate batch loss and accuracy
        # Write training set summary
        summary_str, loss, error, lr, predictions = sess.run([self.summary_op, 
                                                            self.net.locali_cost, 
                                                            self.net.locali_error,
                                                            self.learning_rate_node,
                                                            self.net.locali_predicter], 
                                                            feed_dict={self.net.x: batch_x,
                                                                      self.net.locali_y: batch_locali_y,
                                                                      self.net.keep_prob: 1.})

        pred10 = sess.run([self.net.locali_accuracy], feed_dict={self.net.x: batch_x,
                                                                self.net.locali_y: batch_locali_y,
                                                                self.net.threshold:tf.constant(10),
                                                                self.net.keep_prob: 1.})

        pred15 = sess.run([self.net.locali_accuracy], feed_dict={self.net.x: batch_x,
                                                                self.net.locali_y: batch_locali_y,
                                                                self.net.threshold:tf.constant(15),
                                                                self.net.keep_prob: 1.})

        pred20 = sess.run([self.net.locali_accuracy], feed_dict={self.net.x: batch_x,
                                                                self.net.locali_y: batch_locali_y,
                                                                self.net.threshold:tf.constant(20),
                                                                self.net.keep_prob: 1.})

        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy10", simple_value=pred10),])
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy15", simple_value=pred15),])
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy20", simple_value=pred20),])
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        train_summary_writer.add_summary(summary_str, step)
        train_summary_writer.flush()
        #Precision= {:.4f}, Recall= {:.4f}, 
        logging.info("TRAINING: Iter {:}, Minibatch Loss= {:.8f}, accuracy10= {:.8f}, accuracy15= {:.8f}, accuracy20= {:.8f} Minibatch RMSE error= {:.8f}%".format(step,
                                                                                        loss,
                                                                                        pred10,
                                                                                        pred15,
                                                                                        pred20,
                                                                                        error))
        return pred10, pred15, pred20


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and labels.
    """
    return 100.0 - (100*(np.sum(predictions == labels)/float(predictions.shape[0]*predictions.shape[1])))

def accu(predictions, labels):
    return (100*(np.sum(predictions == labels)/float(predictions.shape[0]*predictions.shape[1])))
    

