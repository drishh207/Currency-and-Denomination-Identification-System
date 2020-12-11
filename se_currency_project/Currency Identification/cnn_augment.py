# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:54:50 2020

@author: Drushti
"""
# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

#parameters for augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 30,
                                   zoom_range = 0.3,
                                   rotation_range = 20,
                                   brightness_range = (0.8,1.5),
                                   width_shift_range = [-10, 0, 10])

#folder directory path
path = 'F:/Currency Identification'
#path where images need to be saved
path1 = 'F:/Currency Identification/training_set/India'
i = 0
for batch in train_datagen.flow_from_directory('pic', target_size=(224,224),
    class_mode='categorical', shuffle=False, batch_size=10,
    save_to_dir=path1, save_prefix='pic'):

    i += 1
    if i > 90: # save 90 images
        break  # otherwise the generator would loop indefinitely
