import h5py
import numpy

class BaseBatchFile(object):
    """
        Class to generate new batches to feed the data to our model during trainning.
        Provide functions to:
            - extract images and labels of a certain number, the batch size, out of the full dataset.
            - keep track of the current index and number of epochs
            - shuffle the data once every epoch.
    """

    def __init__(self, images=None, labels=None, labels_c = None,min_idx=None, max_idx=None,
                 input_file=None, file_path="", img_key=u'train_img',labels_key_c='train_class_labels',
                    labels_key = u'train_loc_labels'):
        """
        :param images: Every image from the video.
        :param labels: Labes for each item in 'image'.
        """
        self._file_path = file_path
        self._input_file = input_file
        self.img_key = img_key
        self.labels_key = labels_key
        self.labels_key_c = labels_key_c
        self._file = input_file
        self._min_idx = min_idx
        self._max_idx = max_idx

        if file_path or input_file:
            self.init_from_file(img_key, labels_key, labels_key_c)
        else:
            self._images = images
            self._labels = labels
            self._labels_c = labels_c

        if not min_idx:
            self._min_idx = 0
        if not max_idx:
            self._max_idx = self._labels.shape[0]


        self._num_examples = self._max_idx - self._min_idx

        self._permutation_vect = None
        self._current_epoch = 0
        self._data_current_idx = 0

        self._current_batch = None

        self.shuffle()

    @property
    def min_idx(self):
        return self._min_idx

    @property
    def max_idx(self):
        return self._max_idx

    @property
    def file(self):
        return self._file

    @property
    def file_path(self):
        return self._file_path
    @property
    def images(self):
        if self.file:
            return numpy.array(self._images[numpy.sort(self._permutation_vect), :])
        else:
            return numpy.array(self._images[self._permutation_vect])

    @property
    def labels(self):
        if self.file:
            return numpy.array(self._labels[numpy.sort(self._permutation_vect), :])
        else:
            return numpy.array(self._labels[self._permutation_vect])
        
    @property
    def labels_c(self):
        if self.file:
            return numpy.array(self._labels_c[numpy.sort(self._permutation_vect), :])
        else:
            return numpy.array(self._labels_c[self._permutation_vect])

    @property
    def num_examples(self):
        """
            Quantity of images in the original dataset (loaded file).
        """
        return self._num_examples#self._permutation_vect.shape[0]


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

    def init_from_file(self, img_key, labels_key, labels_key_c):
        if not self._file:
            self._file = h5py.File(self.file_path, 'r')
        self._images = self._file[img_key]
        self._labels = self._file[labels_key]
        self._labels_c = self._file[labels_key_c]

    def close_file(self):
        self._file.close()

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
            if rest_num_examples > 0:
                try:
                    images_rest_part = self._images[numpy.sort(self._permutation_vect[start:]), :]
                    labels_rest_part = self._labels[numpy.sort(self._permutation_vect[start:]), :]
                    labels_c_rest_part = self._labels_c[numpy.sort(self._permutation_vect[start:]), :]
                except:
                    pass
            else:
                images_rest_part = numpy.empty((0,) + self._images.shape[1:])
                labels_rest_part = numpy.empty((0,) + self._labels.shape[1:])
                labels_c_rest_part = numpy.empty((0,) + self._labels_c.shape[1:])

            self.shuffle()
            start = 0
            self._data_current_idx = batch_size - rest_num_examples
            end = self._data_current_idx
            images_new_part = self._images[numpy.sort(self._permutation_vect[start:end]), :]
            labels_new_part = self._labels[numpy.sort(self._permutation_vect[start:end]), :]
            labels_c_new_part = self._labels_c[numpy.sort(self._permutation_vect[start:end]), :]

            self._current_batch = numpy.concatenate((images_rest_part, images_new_part), axis=0),\
                                  numpy.concatenate((labels_rest_part, labels_new_part), axis=0),\
                                  numpy.concatenate((labels_c_rest_part, labels_c_new_part), axis=0)
            return self._current_batch

        else:
            self._data_current_idx += batch_size
            end = self._data_current_idx

            self._current_batch = numpy.array(self._images[numpy.sort(self._permutation_vect[start:end]), :]), \
                              numpy.array(self._labels[numpy.sort(self._permutation_vect[start:end]), :]), \
                              numpy.array(self._labels_c[numpy.sort(self._permutation_vect[start:end]), :])

            return self._current_batch

    def shuffle(self):
        self._permutation_vect = numpy.arange(self.min_idx, self.max_idx)
        numpy.random.shuffle(self._permutation_vect)
        # self._permutation_vect = numpy.sort(self._permutation_vect)

    def close(self):
        if self.file_path and self.file:
            self.close_file()

class Batch(BaseBatchFile):
    def __init__(self, file_path, images=None, labels=None, labels_c = None,test_perc=0.0, validation_perc=0.0,
                 img_key=u'train_img', labels_key=u'train_loc_labels',labels_key_c='train_class_labels'):
        super(Batch, self).__init__(file_path=file_path, images=images, labels=labels, labels_c = labels_c,
                                    img_key=img_key, labels_key=labels_key, labels_key_c = labels_key_c)

        self.test_percentage = test_perc
        self.validation_percentage = validation_perc
        self.shuffle()

        #initializing
        self.train = BaseBatchFile(images=numpy.zeros((0)), labels=numpy.zeros((0)),labels_c=numpy.zeros((0)))
        self.test = BaseBatchFile(images=numpy.zeros((0)), labels=numpy.zeros((0)),labels_c=numpy.zeros((0)))
        self.validation = BaseBatchFile(images=numpy.zeros((0)), labels=numpy.zeros((0)),labels_c=numpy.zeros((0)))

        size_test = int(numpy.floor(self._labels.shape[0] * test_perc))
        size_validation = int(numpy.floor(self._labels.shape[0] * validation_perc))
        assert size_test + size_validation < self._labels.shape[0]

        if test_perc > 0.0:
            self.test = BaseBatchFile(input_file=self.file, images=images, labels=labels,labels_c = labels_c, min_idx=0, max_idx=size_test)
        if validation_perc > 0.0:
            self.validation = BaseBatchFile(input_file=self.file, images=images, labels=labels, labels_c = labels_c,min_idx=size_test, max_idx=size_test+size_validation)

        if test_perc + validation_perc < 1.0:
            self.train = BaseBatchFile(input_file=self.file,  images=images, labels=labels,labels_c = labels_c,min_idx=size_test+size_validation)

    @property
    def current_epoch(self):
        """
            Number of times the whole train data was used.
            First index: 0.
        """
        return self.train.current_epoch

if __name__ == "__main__":
    # dataset_batch = Batch(file_path='../data/others/train_images_position_gaussian_224_10pct.hdf5', test_perc=0.9, validation_perc=0.1)
    dataset_batch = Batch(file_path='../MLMI_FCN//data/location/ibrar/train_images_224.hdf5')
    dataset_batch_test = Batch(file_path='../MLMI_FCN//data/location/ibrar/test_images_224.hdf5', test_perc=0.9,
                                validation_perc=0.1)

    dataset_batch.test = dataset_batch_test.test
    dataset_batch.validation = dataset_batch_test.validation
    print (dataset_batch.train.labels.shape)
    print (dataset_batch.validation.labels.shape)
    print (dataset_batch.test.labels.shape)
    for idx in range(20):
        dataset_batch.train.next_batch(50)
        print(dataset_batch.train.current_epoch)




"""
# Test (comment the shuffle for sanity test)
# images = 10* np.random.rand(30,1)
# images = images.astype(int)
images = numpy.zeros(shape=(10, 3,2,1))
#print(images)
# labels = np.array([2,4,1,4,7,5,3,2,4,6,6,3,5,2,5,5,2,4,1,4,7,5,3,2,4,6,6,3,5,2])
labels = numpy.array([1,2,3,4,5,6,7,8,9,10])
batch = Batch(file_path=None, images=images, labels=labels, test_perc=0.2, validation_perc=0.3)
print ("Batch generated")
print ("Test: ", batch.test.images.shape)
print ("Validation: ", batch.validation.images.shape)
print ("Train: ", batch.train.images.shape)

print ("DATA:")
print ("Test: ", batch.test.labels)
print ("Validation: ", batch.validation.labels)
print ("Train: ", batch.train.labels)


# print(batch.next_batch(12))
# print(batch.next_batch(12))
# print(batch.next_batch(12))
# print(batch.data_current_idx)
"""
