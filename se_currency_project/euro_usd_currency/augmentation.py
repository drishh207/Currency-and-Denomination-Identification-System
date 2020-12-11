# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:50:22 2020

@author: Drishti
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 30,
                                   zoom_range = 0.3,
                                   rotation_range = 20,
                                   brightness_range = (0.75,1.35),
                                   width_shift_range = [-10, 0, 10])

path = 'E:/Currency_project'
path1 = 'E:/Currency_project/test_set/500'
i = 0
for batch in train_datagen.flow_from_directory('pic', target_size=(224,224),
    class_mode='categorical', shuffle=False, batch_size=10,
    save_to_dir=path1, save_prefix='pic'):

    i += 1
    if i > 12: # save 20 images
        break  # otherwise the generator would loop indefinitely7