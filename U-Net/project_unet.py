
# coding: utf-8

# In[1]:


from tf_unet_modified import unet
from tf_unet_modified.Trainer import Trainer
from tf_unet_modified.Trainer import error_rate
import tensorflow as tf
from BatchGeneration import Batch
from DataPreparation import DataPreparation
from utils import *
import os
import tables
import numpy as np
from copy import deepcopy
import cv2
import time
import matplotlib.pyplot as plt


# IMPORT DATA

# In[ ]:


#input_videos_dir = "data/Tracking_Robotic_Training/Training/"
#input_labels_dir = "data/EndoVisPoseAnnotation-master/train_labels"
#output_dir = "data/Tracking_Robotic_Training/Training_set/"
#output_file_name = "train_images_128.hdf5"


# In[ ]:


#dp = DataPreparation(input_videos_dir,input_labels_dir, output_dir)
#print("------------------Preparing Data--------------------- ")
#dp.extractFrames(output_file_name)


# In[2]:



#To check training set
trainhdf5filename = "data/Tracking_Robotic_Training/Training_set/train_images_224_10_joints.hdf5"
testhdf5filename = "data/Test_set/test_images_224_10_joints.hdf5"
"""
# Train image show
trainhdf5file = os.path.abspath(trainhdf5filename)
trainhdf5file = tables.open_file(trainhdf5file, "r")
X = trainhdf5file.root.images
labs = trainhdf5file.root.rawlabels
"""



# Class declaration for handling the u-net training, 

# In[3]:


class BatchGenerator:
    def __init__(self, train_images, train_classi_labels, train_locali_labels, test_images, test_classi_labels,
                 test_locali_labels):
        self._timages = train_images
        self._tclabels = train_classi_labels
        self._tllabels = train_locali_labels
        self._testimages = test_images
        self._testclabels = test_classi_labels
        self._testllabels = test_locali_labels

        self.train_offset = 0
        self.val_offset = 0
        self.test_offset= 0
        
        self.tr_size = self._timages.shape[0]
        self.test_size = self._testimages.shape[0]
        self.val_size = int(np.floor(self.test_size * 0.3)) # Use 30% of test set as validation
        
        self.indices_train = np.arange(self.tr_size)
        self.indices_test = np.arange(self.test_size)

        np.random.shuffle(self.indices_train)
        np.random.shuffle(self.indices_test)
        
        self.indices_val = self.indices_test[0:self.val_size]
        self.indices_test = self.indices_test[self.val_size:]
        self.test_size = len(self.indices_test)

    def get_next_training_batch(self, batch_size):
        images = []
        clabels = []
        llabels = []
        if len(self.indices_train) >= batch_size:
            images = self._timages[self.indices_train[0:batch_size],:]
            clabels = self._tclabels[self.indices_train[0:batch_size],:]
            llabels = self._tllabels[self.indices_train[0:batch_size],:]
            self.indices_train = self.indices_train[batch_size:]
        else:
            images = self._timages[self.indices_train[0:],:]
            clabels = self._tclabels[self.indices_train[0:],:]
            llabels = self._tllabels[self.indices_train[0:],:]
            self.indices_train = []
            
        if len(self.indices_train) == 0:
            print("Training epoch happening after this return")
            # An epoch happened on val data
            # shuffle VALIDATION data indices
            self.indices_train = np.arange(self.tr_size)
            np.random.shuffle(self.indices_train)
        return images, clabels, llabels
    
    def get_next_validation_batch(self, batch_size):
        images = []
        clabels = []
        tlabels = []
        if (self.val_offset + batch_size) <= len(self.indices_val):
            images = self._testimages[self.indices_val[self.val_offset:self.val_offset+batch_size], :]
            clabels = self._testclabels[self.indices_val[self.val_offset:self.val_offset+batch_size], :]
            llabels = self._testllabels[self.indices_val[self.val_offset:self.val_offset+batch_size], :]
            self.val_offset += batch_size
        else:
            images = self._testimages[self.indices_val[self.val_offset:],:]
            clabels = self._testclabels[self.indices_val[self.val_offset:],:]
            llabels = self._testllabels[self.indices_val[self.val_offset:],:]
            self.val_offset += batch_size
            
        if self.val_offset >= len(self.indices_val):
            print("Validation epoch happening after this return")
            # An epoch happened on val data
            # shuffle VALIDATION data indices
            np.random.shuffle(self.indices_val)
            self.val_offset = 0
        return images, clabels, llabels
    
    def get_next_test_batch(self, batch_size):
        images = []
        clabels = []
        tlabels = []
        if (self.test_offset + batch_size) <= len(self.indices_test):
            images = self._testimages[self.indices_test[self.test_offset:self.test_offset+batch_size], :]
            clabels = self._testclabels[self.indices_test[self.test_offset:self.test_offset+batch_size], :]
            llabels = self._testllabels[self.indices_test[self.test_offset:self.test_offset+batch_size], :]
            self.test_offset += batch_size
        else:
            images = self._testimages[self.indices_test[self.test_offset:],:]
            clabels = self._testclabels[self.indices_test[self.test_offset:],:]
            llabels = self._Testllabels[self.indices_test[self.test_offset:],:]
            self.test_offset += batch_size
            
        if self.test_offset >= len(self.indices_test):
            print("Test epoch happening after this return")
            # An epoch happened on test data
            # shuffle TEST data indices
            np.random.shuffle(self.indices_test)
            self.test_offset = 0
        return images, clabels, llabels

class DataProvider:
    """
    Class which provides data related functions
    """

    def __init__(self, trainhdf5file, testhdf5file):
        trainhdf5file = os.path.abspath(trainhdf5file)
        testhdf5file = os.path.abspath(testhdf5file)

        trainhdf5file = tables.open_file(trainhdf5file, "r")
        testhdf5file = tables.open_file(testhdf5file, "r")
        print(trainhdf5file)
        print(testhdf5file)
        
        self.train_X = np.array(trainhdf5file.root.images)/255.0
        self.train_classi_Y = np.array(trainhdf5file.root.classilabels)
        self.train_locali_Y = np.array(trainhdf5file.root.localilabels)
        
        self.test_X = np.array(testhdf5file.root.images)/255.0
        self.test_classi_Y = np.array(testhdf5file.root.classilabels)
        self.test_locali_Y = np.array(testhdf5file.root.localilabels)
    
        # To make everything fast at the expense of huge RAM usage, pass these handlers as numpy arrays 
        # to BatchGenerator
        self.batch_handler = BatchGenerator(self.train_X, self.train_classi_Y, self.train_locali_Y,
                                            self.test_X, self.test_classi_Y, self.test_locali_Y)

    def get_training_batch(self, n):
        return self.batch_handler.get_next_training_batch(n)
    
    def no_validation_batches(self, batch_size):
        if len(self.batch_handler.indices_val)%batch_size == 0:
            return len(self.batch_handler.indices_val)/batch_size
        else:
            return len(self.batch_handler.indices_val)/batch_size + 1
    
    def no_training_batches(self, batch_size):
        if len(self.batch_handler.indices_train)%batch_size == 0:
            return len(self.batch_handler.indices_train)/batch_size
        else:
            return len(self.batch_handler.indices_train)/batch_size + 1
    
    def no_test_batches(self, batch_size):
        if len(self.batch_handler.indices_test)%batch_size == 0:
            return len(self.batch_handler.indices_test)/batch_size
        else:
            return len(self.batch_handler.indices_test)/batch_size + 1
            
    def get_validation_batch(self, n):
        return self.batch_handler.get_next_validation_batch(n)

    def get_test_batch(self, n):
        return self.batch_handler.get_next_test_batch(n)

# In[ ]:


# Test the code below

# In[5]:

"""
# Test data preparator
data_provider = DataProvider(trainhdf5filename, testhdf5filename)
tr_X, tr_classi, tr_locali = data_provider.get_training_batch(50)
te_X, te_classi, te_locali = data_provider.get_validation_batch(50)


plt.imshow(tr_X[20])
plt.show()

print(tr_X.shape)
print(tr_classi.shape)
print(tr_locali.shape)

print ("Print train images now")

for i in range(tr_X.shape[0]):
    gray = tr_X[i]
    new_pmaps = tr_locali[i]
    for joint in range(new_pmaps.shape[0]):
        x,y = np.unravel_index(new_pmaps[joint,:,:].argmax(),[224,224])
        cv2.circle(gray,(int(y),int(x)), 5, (0,255,0), -1)
    print(tr_classi[i])
    plt.imshow(gray)
    plt.show()

print ("Print test images now")
print(te_X.shape)
print(te_classi.shape)
print(te_locali.shape)
for i in range(te_X.shape[0]):
    gray = te_X[i]
    new_pmaps = te_locali[i]
    for tool in range(new_pmaps.shape[3]):
        for joint in range(new_pmaps.shape[2]):
            x,y = np.unravel_index(new_pmaps[:,:,joint,tool].argmax(),[224,224])
            cv2.circle(gray,(int(y),int(x)), 5, (0,255,0), -1)
    print(te_classi[i])
    plt.imshow(gray)
    plt.show()
"""


# In[10]:



# In[ ]:


#preparing data loading
data_provider = DataProvider(trainhdf5filename, testhdf5filename)
#print(data_provider.train_X[0])
#plt.imshow(data_provider.train_X[0])

#plt.show()

#setup & training
# Initial channels are 3, initial output features expected are 64, 2 classes and forming 3 layers 
# where each layer does 2 convolutions with RELU and one pooling

net = unet.Unet(layers=6, features_root=64, channels=3, n_class=4, cost="sigmoid_cross_entropy", lambda_c = 1, lambda_l = 1)

trainer = Trainer(net, batch_size=10, optimizer="adam", opt_kwargs={"learning_rate":5e-04})

model_path = trainer.train(data_provider, "model_saved", training_iters= 380,epochs=40, write_graph=True) #95*40 = 3800 we have 3760 images

"""
test_X, test_classi_Y, test_locali_Y = data_provider.get_test_batch(10)
image = test_X[0]
#import matplotlib.pyplot as plt 
plt.imshow(image)
plt.show()
prediction_x, prediction_y = net.locali_predict("model_saved", test_X)
print(prediction_x.shape)
print(prediction_y.shape)
for i in range(test_X.shape[0]):
    gray = test_X[i]
    for j in range(prediction_x.shape[1]):
        print(prediction_y[i,j])
        cv2.circle(gray, (int(prediction_y[i,j]), int(prediction_x[i,j])), 5, (0,255,0), -1)
    plt.imshow(gray)
    plt.show()


#verification
# Let's test on the test set
print("test images")
testfilename = "data/test_images.hdf5"
testfile = os.path.abspath(testfilename)
testfile = tables.open_file(testfile, "r")
X = testfile.root.train_img
labs = testfile.root.train_labels
test_Y = []
for image_label in labs:
    hot_vector = [0,0,0,0] # We are assuming 4 kinds of instruments can be present in the image
    if 1111 in image_label:
        # Means that right clasper instrument is present
        # Change the label at hot_vector[0] to 1
        hot_vector[0] = 1
    if 1110 in image_label:
        # Means that left clasper instrument is present
        # Change the label at hot_vector[1] to 1
        hot_vector[1] = 1
    if 1100 in image_label:
        # Means that right scissor instrument is present
        # Change the label at hot_vector[2] to 1
        hot_vector[2] = 1
    if 1000 in image_label:
        # Means that left scissor instrument is present
        # Change the label at hot_vector[3] to 1
        hot_vector[3] = 1
    test_Y.append(hot_vector)
test_Y = np.array(test_Y)
print(X.shape)
print(test_Y.shape)

image = X[0]
#import matplotlib.pyplot as plt 
#plt.imshow(image)
#plt.show()
batches = 0
if test_Y.shape[0]%50 == 0:
    batches = test_Y.shape[0]/50
else:
    batches = test_Y.shape[0]/50 + 1
start = 0
error = 0
for i in range(batches):
    prediction = net.predict("model_saved", X[start:start+50])
    error += error_rate(prediction, test_Y[start:start+50])
    start += 50
    
error /= float(batches)
print("error is:")
print(error)
    #print("real")
    #print(test_Y[900:950])
    #print("prediction")
    #print(prediction)
#print(data_provider.batch_handler.labels_test[10:15])
#print(error_rate(prediction, data_provider.batch_handler.labels_test[10:15]))

#mg = util.combine_img_prediction(self.batch_handler.images_test, self.batch_handler.labels_test, prediction)
#til.save_image(img, "prediction.jpg")
"""


