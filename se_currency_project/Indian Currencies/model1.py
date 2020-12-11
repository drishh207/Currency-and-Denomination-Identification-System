# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:29:13 2020

@author: Drushti
"""

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.applications import VGG19 #For Transfer Learning
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.utils import to_categorical

EPOCHS = 12
BS = 100 #Batch size
LR = 1e-3 #Learning rate 0.001
img_dim = (45,45,3)
train_data_dir = 'F:/Currency Identification/training_set'
test_data_dir = 'F:/Currency Identification/test_set'
labels = []
#Nbr of training images
train_samples_nbr  = sum(len(files) for _, _, files in os.walk(r'F:/Currency Identification/training_set'))
#Nbr of testing images
test_samples_nbr  = sum(len(files) for _, _, files in os.walk(r'F:/Currency Identification/test_set'))


nbr_of_pictures = []
labels = os.listdir("F:/Currency Identification/training_set")
for _, _, files in os.walk(r'F:/Currency Identification/training_set'):
    nbr_of_pictures.append(len(files))

nbr_of_pictures=nbr_of_pictures[1:]
#print nbr of pictures in every class
print("Number of samples in every class ...")
for i in range(3):  # 82 : Nbr of classes
    print(labels[i]," : ",nbr_of_pictures[i])
    

