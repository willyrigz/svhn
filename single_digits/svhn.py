#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Aspire E15
#
# Created:     25-05-2019
# Copyright:   (c) Aspire E15 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os
import sys
import time
import tarfile
from IPython.display import clear_output
from PIL import Image, ImageDraw
import pandas as pd
import urllib.request

import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
import itertools
from keras.utils.np_utils import to_categorical

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from keras import backend as K
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import cv2

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    pass

def rgb2gray(image):
        """Convert images from rbg to grayscale
        """
        return np.expand_dims(np.dot(image, [0.2989, 0.5870, 0.1140]), axis=3)

def test(img):
    print("\nPredicting Number from Image\n")

    img_read = cv2.imread(img, -1)
    if len(img_read.shape) > 2 and img_read.shape[2] == 4:
        #convert the image from RGBA2RGB
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGRA2BGR)
    img_read = cv2.resize(img_read, (32, 32)).astype(np.float32)
    img_test = []

    img_test.append(np.asarray(img_read))
    img_test = np.array(img_test)

    img_test_greyscale = rgb2gray(img_test).astype(np.float32)

    new_model = load_model('single_model.h5')

    score = new_model.predict(img_test_greyscale)

    roundedScore = np.zeros(score.shape, dtype="int32")
    roundedScore[score > 0.5] = 1

    print('Predicted Number: ',np.argmax(roundedScore[0:1,:]))

    prediction = "\nPredicted.\n"
    return(prediction)


def traintest():
    create_path = "data"
    if not os.path.exists(create_path):
        os.mkdir(create_path)
    print('Downloading files.....\n')
    if os.path.isfile("data/train_32x32.mat"):
        print('Train matfile already downloaded')
    else:
        urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "data/train_32x32.mat")
    if os.path.isfile("data/test_32x32.mat"):
        print('Test matfile already downloaded')
    else:
       urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "data/test_32x32.mat")
    if os.path.isfile("data/extra_32x32.mat"):
        print('Extra matfile already downloaded\n')
    else:
        urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", "data/extra_32x32.mat")



    print('files Downloaded\n')


    # In[2]:


    print('Loading MAT files into dictionary....\n')
    train_data = sio.loadmat('data/train_32x32.mat')
    extra_data = sio.loadmat('data/extra_32x32.mat')
    test_data = sio.loadmat('data/test_32x32.mat')


    # In[4]:


    # access to the dict
    print('Access dataset from dictionary....\n')
    x_train = train_data['X']
    y_train = train_data['y']
    x_extra = extra_data['X']
    y_extra = extra_data['y']
    x_test = test_data['X']
    y_test = test_data['y']

    # In[6]:


    # Transpose the image arrays
    X_train, y_train = x_train.transpose((3,0,1,2)), y_train[:,0]
    X_extra, y_extra = x_extra.transpose((3,0,1,2)), y_extra[:,0]
    X_test, y_test = x_test.transpose((3,0,1,2)), y_test[:,0]


    # In[7]:


    print("Training Set", X_train.shape)
    print("Extra Set", X_extra.shape)
    print("Test Set", X_test.shape)
    print('')


    # In[21]:


    print('Changing image label for number 0 from 10 to 0\n')
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    y_extra[y_extra == 10] = 0


    # Transform the images to greyscale
    print("Converting all images to Grayscale (Makes it quicker for training and stuff)....\n")
    train_greyscale = rgb2gray(X_train).astype(np.float32)
    test_greyscale = rgb2gray(X_test).astype(np.float32)
    extra_greyscale = rgb2gray(X_extra).astype(np.float32)

    print("Dimensions")
    print("Training set", train_greyscale.shape)
    print("Test set", test_greyscale.shape)
    print("Extra set", extra_greyscale.shape)
    print('')


    # In[18]:


    print("Normalize all images to be within 0 and 1 by dividing by 255...\n")

    train_greyscale /= 255
    test_greyscale /= 255
    extra_greyscale /= 255


    # In[22]:


    print("Categorise labels for model...\n")

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_extra = to_categorical(y_extra, 10)

    print(y_train.shape)
    print(y_test.shape)
    print(y_extra.shape)


    # In[ ]:


    print('Randomly collect 20% from both training and extra dataset each and combine to create a validation set\n')
    print('Combine remaining 80% in training and extra to make full training set for model\n')

    X_test = test_greyscale.copy()
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(train_greyscale, y_train, test_size=0.20, shuffle= True, random_state=66)
    X_extra_train, X_extra_val, y_extra_train, y_extra_val = train_test_split(extra_greyscale, y_extra, test_size=0.20, shuffle= True, random_state=66)

    X_train = np.concatenate([X_train_train, X_extra_train])
    y_train = np.concatenate([y_train_train, y_extra_train])

    X_val = np.concatenate([X_train_val, X_extra_val])
    y_val = np.concatenate([y_train_val, y_extra_val])

    print('Full Training set', X_train.shape, y_train.shape)
    print('Full Validation set', X_val.shape, y_val.shape)


    # In[ ]:


    a = Input(shape=(32, 32, 1))
    #conv layer
    c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(a)
    c12 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
    c1p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c12)
    c1d = Dropout(rate=0.1)(c1p)

    c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1d)
    c22 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
    c2p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c22)
    c2d = Dropout(rate=0.1)(c2p)

    c3 = Conv2D(128, (3, 3), padding='same', activation='relu')(c2d)
    c32 = Conv2D(128, (3, 3), padding='same', activation='relu')(c3)
    c33 = Conv2D(128, (3, 3), padding='same', activation='relu')(c32)
    c3p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c33)
    c3d = Dropout(rate=0.1)(c3p)

    c4 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3d)
    c42 = Conv2D(256, (3, 3), padding='same', activation='relu')(c4)
    c43 = Conv2D(256, (3, 3), padding='same', activation='relu')(c42)
    c4p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c43)
    c4d = Dropout(rate=0.1)(c4p)

    c5 = Conv2D(512, (3, 3), padding='same', activation='relu')(c4d)
    c52 = Conv2D(512, (3, 3), padding='same', activation='relu')(c5)
    c53 = Conv2D(512, (3, 3), padding='same', activation='relu')(c52)
    c5p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c53)
    c5d = Dropout(rate=0.1)(c5p)


    #linear layer
    flat = Flatten()(c5d)
    d1 = Dense(1024, activation='relu')(flat)
    d1d = Dropout(rate=0.1)(d1)
    d2 = Dense(1024, activation='relu')(d1d)
    d2d = Dropout(rate=0.1)(d2)

    #output
    o_digit = Dense(10, activation='softmax')(d2d)

    model = Model(a, o_digit)

    model.compile(Adam(lr=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    print('Fit Model..........................\n')
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=1000)


    # In[ ]:


    print('Test Model\n')
    #model = load_model('single_model.h5')

    #validation
    score = model.predict(X_test)

    #score = np.concatenate(score, axis=1)
    #round score to get actual prediction instead of probability
    roundedScore = np.zeros(score.shape, dtype="int32")
    #choose threshold
    roundedScore[score > 0.5] = 1

    #rounded acc (exactly match)
    print('Test Accuracy: ', accuracy_score(y_test, roundedScore))

    avg_f1 = f1_score(y_test.argmax(axis=1), roundedScore.argmax(axis=1), average='macro')
    print('Macro Avg F1 score: ', avg_f1)
    report = classification_report(y_test.argmax(axis=1), roundedScore.argmax(axis=1))
    return report



if __name__ == '__main__':
    main()
