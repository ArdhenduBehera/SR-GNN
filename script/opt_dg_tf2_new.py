"""
Directory based data generator supporing image augmentation and pre-processing for multiple data streams.
Augmentation is identical across all streams (by passing a keras.preprocessing.image.ImageDataGenerator instance with the desired parameters),
but pre-processing is individual.

Created by Alexander Keidel @ 17.05.2018
Modified by Andrew Gidney @ 10.08.2018
"""

import numpy as np
from os import listdir
from os.path import isdir, join, isfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import cv2
import random

class DirectoryDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_directories, augmentor=False, preprocessors=None, batch_size=16, target_sizes=(224,224), shuffle=False, channel_last=True, hasROIS=False, verbose=True):
        self.base_directories = base_directories
        self.augmentor = augmentor
        self.preprocessors = preprocessors #should be a function that can be directly called with an image 
        self.batch_size = batch_size
        self.target_sizes = target_sizes
        self.shuffle = shuffle
        
        self.class_names = []
        self.files = []
        self.labels = []
        #self.quick_batch = quick_batch
        
        for base_directory in base_directories:
            class_names = [x for x in listdir(base_directory)  if isdir(join(base_directory, x))] #only folders are added
            class_names = sorted(class_names)
            self.class_names.append(class_names)
            
            for i, c in enumerate(class_names):
                class_dir = join(base_directory, c)
                if isdir(class_dir):
                    #here we can add the class names if we wanted too as well by making a list
                    for f in listdir(class_dir):
                        file_dir = join(class_dir, f)
                        if (isfile(file_dir) and file_dir.lower().endswith((".jpg", ".png"))):
                            self.files.append(file_dir)
                            self.labels.append(i)

        if verbose:
            for i in range(len(self.class_names[0])):
                lbls = []
                for c in self.class_names:
                    lbls.append(c[i])
                # print('Using label {} for class_names: {}'.format(i, lbls)) # ~~~ <<< print turned off >>

        self.nb_classes = len(self.class_names[0])
        self.nb_files = len(self.files)
        self.on_epoch_end()
        
        if verbose:
            print('Found {} images for {} classes.'.format(self.nb_files, self.nb_classes))
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nb_files / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, indexes)
        return [X,], y

    def get_indexes(self):
        return self.indexes

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nb_files)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def cv2_image_augmentation(self, img, theta=20, tx=10, ty=10, scale=1., flip=False):
        
        if scale != 1.:
            scale = np.random.uniform(1-scale, 1+scale)
            
        if theta != 0:
            theta = np.random.uniform(-theta, theta)    
        
        m_inv = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), theta, scale)
        
        if tx != 0 or ty != 0:
            tx = np.random.uniform(-tx, tx)
            ty = np.random.uniform(-ty, ty)
            m_inv[0,2] += tx
            m_inv[1,2] += ty
        
        image = cv2.warpAffine(img, m_inv, (img.shape[1], img.shape[0]), borderMode=1)
        if flip:            
            if round(random.random()):
                image = np.fliplr(image)
        return image

    def __data_generation(self, list_IDs_temp, indexes):
        # Initialization
        X = np.empty((self.batch_size, self.target_sizes[0],self.target_sizes[1], 3), dtype = K.floatx())
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            img = cv2.imread(ID)
            img = cv2.resize(img, self.target_sizes)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ###Swaps from opencv colour order to skimage order (might produce better outputs on some datasets)
            img = img.astype(K.floatx())
            
            if self.augmentor:
                img = self.cv2_image_augmentation(img, theta=15, tx=0., ty=0., scale=0.15, flip=False)          
            
            if self.preprocessors:                   
                img = self.preprocessors(img)  
            
            X[i,] = img
            y[i] = self.labels[indexes[i]] 
                
        return X, tf.keras.utils.to_categorical(y, num_classes=self.nb_classes)