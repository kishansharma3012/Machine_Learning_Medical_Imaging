# coding: utf-8
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class BaseBatch(object):
    """
        Class to generate new batches to feed the data to our model during trainning.
        Provide functions to:
            - extract images and labels of a certain number, the batch size, out of the full dataset.
            - keep track of the current index and number of epochs
            - shuffle the data once every epoch.
    """

    def __init__(self, images, labels):
        """
        :param images: Every image from the video.
        :param labels: Labes for each item in 'image'.
        """

        self._images = images
        self._labels = labels
        self._current_epoch = 0
        self._data_current_idx = 0
        self._num_examples = images.shape[0]
        self._current_batch = None

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        """
            Quantity of images.
        """
        return self._num_examples

    @property
    def current_epoch(self):
        """
            Number of times the whole data was used.
            First index: 0.
        """
        return self._current_epoch

    @property
    def data_current_idx(self):
        """
            Index of image in the current epoch.
        """
        return self._data_current_idx

    @property
    def current_batch(self):
        """
            Current batch of image and labels.
        """
        return self._current_batch

    def next_batch(self, batch_size):
        """
        :param batch_size: represents the step size.
        :return: the next 'batch_size' examples from this data set (images[start:end], labels[start:end]).
        """
        start = self._data_current_idx

        if self._current_epoch == 0 and start == 0:
            self.shuffle()

        if start + batch_size > self._num_examples:
            self._current_epoch += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            self.shuffle()
            start = 0
            self._data_current_idx = batch_size - rest_num_examples
            end = self._data_current_idx
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            self._current_batch = np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
            return self._current_batch

        else:
            self._data_current_idx += batch_size
            end = self._data_current_idx
            self._current_batch = self._images[start:end], self._labels[start:end]

            return self._current_batch

    def shuffle(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]

class Batch(BaseBatch):
    
    def __init__(self, images, labels, test_perc=0.0, validation_perc=0.0):
        super(Batch, self).__init__(images, labels)

        self.test_percentage = test_perc
        self.validation_percentage = validation_perc
        self.shuffle()
        self.test = BaseBatch(np.zeros((0)),np.zeros((0)))
        self.validation = BaseBatch(np.zeros((0)),np.zeros((0)))

        size_test = int(np.floor(labels.shape[0] * test_perc))
        size_validation = int(np.floor(labels.shape[0] * validation_perc))
        assert size_test + size_validation < labels.shape[0]


        if test_perc > 0.0:
            self.images_test = self.images[0:size_test]
            self.labels_test = self.labels[0:size_test]
            self.test = BaseBatch(self.images_test, self.labels_test)
        if validation_perc > 0.0:
            self.images_validation = self.images[size_test:size_test+size_validation]
            self.labels_validation = self.labels[size_test:size_test+size_validation]
            self.validation = BaseBatch(self.images_validation, self.labels_validation)

        images_train = self.images[size_test+size_validation:]
        labels_train = self.labels[size_test+size_validation:]
        self.train = BaseBatch(images_train, labels_train)


    @property
    def current_epoch(self):
        """
            Number of times the whole train data was used.
            First index: 0.
        """
        return self.train.current_epoch
        

if __name__ == "__main__":
    # The code below will be run only if this file is directly executed from
    # a python interpreter. No need to comment the code out.

    # Test (comment the shuffle for sanity test)
    # images = 10* np.random.rand(30,1)
    # images = images.astype(int)
    images = np.zeros(shape=(10, 1))
    #print(images)
    # labels = np.array([2,4,1,4,7,5,3,2,4,6,6,3,5,2,5,5,2,4,1,4,7,5,3,2,4,6,6,3,5,2])
    labels = np.array([1,2,3,4,5,6,7,8,9,10])
    batch = Batch(images, labels, test_perc=0.1, validation_perc=0.1)
    print ("Batch generated")
    print ("Test: ", batch.test.images.shape)
    print ("Validation: ", batch.validation.images.shape)
    print ("Train: ", batch.train.images.shape)

    print ("Test: ", batch.test.labels)
    print ("Validation: ", batch.validation.labels)
    print ("Train: ", batch.train.labels)


    # print(batch.next_batch(12))
    # print(batch.next_batch(12))
    # print(batch.next_batch(12))
    # print(batch.data_curre
