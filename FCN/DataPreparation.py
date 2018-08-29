#import nibabel as nib
import numpy as np
#import dicom
import matplotlib.pyplot as plt
import time
import os
import re
import h5py
import tables
from tables import *
#from skimage import measure
#from skimage.util.montage import montage2d
#from IPython import display
import cv2
import pdb
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class DataPreparation:
    
    def __init__(self, input_dir, output_dir ):
    
        self.input_dir = os.getcwd() + input_dir
        self.output_dir =os.getcwd() + output_dir
        self.dict_videos = {}
        #pdb.set_trace()
        for sub_dir in next(os.walk(self.input_dir))[1]:
            label_file_list_path = []
            for files in os.listdir(os.path.join(self.input_dir,sub_dir)):
                if files.endswith(".avi"):
                    video_file_path = os.path.join(self.input_dir,sub_dir,files)
                elif files.endswith(".txt"):
                    label_file_list_path.append(os.path.join(self.input_dir,sub_dir,files)) 
            self.dict_videos[video_file_path] = label_file_list_path 

        if not(os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)

    # Returns the sum of frames of all the videos
    def get_no_images(self):
        no_images = 0
        for videofilepath in self.dict_videos:
            print(videofilepath)
            videofilepath = videofilepath
            cap = cv2.VideoCapture(videofilepath)
            no_images = no_images + int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return no_images

    # Extracts the frames and labels and stores in a hdf5 file
    def extractFrames(self, output_file_name):
        
        # Count no of images
        no_images = self.get_no_images()
        print(no_images)

        # Generate absolute file paths for the label files
        for video in self.dict_videos:
            for i in range(len(self.dict_videos[video])):
                self.dict_videos[video][i] = self.dict_videos[video][i]


        # Create hdf5 file
        hdf5_file = tables.open_file(self.output_dir +  '/' + output_file_name, mode='w')

        # Create hdf5 dataset in hdf5_file for training images
        train_store = hdf5_file.create_earray(hdf5_file.root, 'train_img', tables.UInt8Atom(), shape=(0,576,720,3))
        # Create hdf5 dataset in hdf5_file for training labels
        labelarray = hdf5_file.create_vlarray(hdf5_file.root, 'train_labels', tables.Float32Atom(shape=()), "labelarray")

        print("Working.. Wait 2-3 minutes. Or pick the code and parallelize it. I am sure that will be more effort so just wait :)")
        counter = 0
        # Store Images in the file 
        for videofilepath in self.dict_videos:
            videofilepath = videofilepath
            cap = cv2.VideoCapture(videofilepath)
            # Write the images into image file first
            while(cap.isOpened()):
                ret, frame = cap.read()
                sys.stdout.write('\r Frame Extraction Progress : ( %d / %d)' %(counter+1,no_images))           
                sys.stdout.flush()
                counter +=1
                if ret == True:
                    train_store.append(frame[None])
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()

        # Store labels in to the hdf5 file. If an image has two/more instruments,
        # the no of labels for that would be 7*(no. of instruments) stored as a 
        # single sequence in array. Had to do this due to non-uniformity between
        # between no. of instruments in every video. Couldn't find such a way to
        # store non-homogeneous 2-d arrays in hdf5.
        labels_list = [None]*no_images
        base_index = 0
        image_no = 0
        for video in self.dict_videos:
            #print(base_index)
            for labels in self.dict_videos[video]:
                image_no = base_index

                with open(labels) as f:
                    for line in f:
                        #print(image_no)
                        if labels_list[image_no] is None:
                            labels_list[image_no] = []
                        label_data = line.split()
                        labels_list[image_no] = labels_list[image_no] + label_data
                        image_no = image_no + 1
            base_index = image_no

        # Write the labels array to hdf5 file
        for label in labels_list:
            labelarray.append(label)
        hdf5_file.close()


    def data_augmentation(self,input_file):
        """
        Augment the given data : 1) Flip the image left_right

        Input :  path of the folder containing the frames (for e.g training set)
        Output:  New folder with augmented images (for e.g training_set_augment) & new label file 

        """  
        input_file =  self.output_dir + input_file
        input_hdf5_file = tables.open_file(input_file, mode ="r+")
        image_shape = input_hdf5_file.root.train_img.shape
        for i in range(image_shape[0]):
            
            sys.stdout.write('\r Augmentation Progress : ( %d / %d)' %(i+1,image_shape[0]))           
            sys.stdout.flush()
            #cv2.imshow('original_image',hdf5_file.root.train_img[i])
            #cv2.imwrite("/home/kishan/Documents/MLMI_LAB/MLMI/data/Tracking_Robotic_Training/Training_set/original.jpg",hdf5_file.root.train_img[i])
            original_label = input_hdf5_file.root.train_labels[i]
            #print("original _ labels: ",original_label )
            sess = tf.Session()
            flip_image = tf.image.flip_left_right(input_hdf5_file.root.train_img[i])
            flip_image = sess.run(flip_image)
            input_hdf5_file.root.train_img[i] = flip_image


            #cv2.imshow('flip_image',flip_image)
            #cv2.imwrite("/home/kishan/Documents/MLMI_LAB/MLMI/data/Tracking_Robotic_Training/Training_set/flip.jpg",flip_image)

            #center_point
            original_label[0] = image_shape[2] - int(original_label[0]) + 1
            original_label[1] = image_shape[1] - int(original_label[1]) + 1
            #shaft_axis
            original_label[2] = -original_label[2]
            original_label[3] = original_label[3]
            #head_axis
            original_label[4] = -original_label[4]
            original_label[5] = original_label[5]
            #clasper angle
            original_label[6] = original_label[6]

            input_hdf5_file.root.train_labels[i] = original_label

        input_hdf5_file.close() 
        #output_hdf5_file.close()

# <----------------Test the class here---------------->

Input_folder = "/data/Tracking_Robotic_Training/Training/"
Output_folder = "/data/Tracking_Robotic_Training/Training_set/"
output_file_name = "train_images.hdf5"

dp = DataPreparation( Input_folder,Output_folder)

print("------------------Preparing Data--------------------- ")
dp.extractFrames(output_file_name)
print("------------------Augmenting Data--------------------- ")
dp.data_augmentation(output_file_name)
