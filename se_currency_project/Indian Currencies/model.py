# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:08:06 2020

@author: Drushti
"""

# Convolutional Neural Network

# Importing the libraries
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn import metrics 
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                  horizontal_flip = False)


# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)


#Denomination prediction
# Preprocessing the Training set
training_set1 = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set1 = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


#print classes in each class
train_data_dir = 'E:/se_currency_project/Indian Currencies/training_set'
test_data_dir = 'E:/se_currency_project/Indian Currencies/test_set'
labels = []
#Nbr of training images
train_samples_nbr  = sum(len(files) for _, _, files in os.walk(r'E:/se_currency_project/Indian Currencies/training_set'))
#Nbr of testing images
test_samples_nbr  = sum(len(files) for _, _, files in os.walk(r'E:/se_currency_project/Indian Currencies/test_set'))


nbr_of_pictures = []
labels = os.listdir("E:/se_currency_project/Indian Currencies/training_set")
for _, _, files in os.walk(r'E:/se_currency_project/Indian Currencies/training_set'):
    nbr_of_pictures.append(len(files))

nbr_of_pictures=nbr_of_pictures[1:]
#print nbr of pictures in every class
print("Number of samples in every class ...")
for i in range(7):  # 82 : Nbr of classes
    print(labels[i]," : ",nbr_of_pictures[i])
    
    

# Part 2 - Building the CNN

# Initialising the CNN
cnn1 = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn1.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn1.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a third convolutional layer
cnn1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn1.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a fourth convolutional layer
cnn1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn1.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn1.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn1.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer --> sigmoid
cnn1.add(tf.keras.layers.Dense(units=7, activation='softmax'))


# Part 3 - Training the CNN

# Compiling the CNN
cnn1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn1.fit(x = training_set1, validation_data = test_set1, epochs = 40)
cnn1.summary()

cnn1.save('Indian_denomination.h5')

#Plotting the training and validation loss and accuracy
f,ax=plt.subplots(2,1,figsize=(10,5))  #Creates 2 subplots under 1 column

#Loss --> #Assigning the first subplot to graph training loss and validation loss
ax[0].set_title('Model Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].plot(cnn1.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(cnn1.history.history['val_loss'],color='r',label='Validation Loss')
legend = ax[0].legend(loc='best', shadow=True)
#Accuracy --> #Plotting the training accuracy and validation accuracy
#plt.title('Model Accuracy')
ax[1].set_title('Model Accuracy')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].plot(cnn1.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(cnn1.history.history['val_accuracy'],color='r',label='Validation Accuracy')
legend = ax[1].legend(loc='best', shadow=True)

#classification report
test_steps_per_epoch = np.math.ceil(test_set1.samples / test_set1.batch_size)
predictions = cnn1.predict_generator(test_set1, steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_set1.classes
class_labels = list(test_set1.class_indices.keys())   

print("Classification Report")
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)    


#Defining function for confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    #Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#print(cm)

    fig, ax = plt.subplots(figsize=(9,9))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')


    #Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)


class_names = ['10','100','20','200','2000','50','500']

#Plotting normalized confusion matrix
plot_confusion_matrix(test_set1.classes, predicted_classes, classes = class_names, normalize = True, title = 'Normalized confusion matrix')


"""import numpy as np
from keras.preprocessing import image
test_image = image.load_img('predict/20_predict.jpg', target_size = (64, 64))
test_image.show()
#print(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result1 = cnn1.predict(test_image)
print(result1)"""


