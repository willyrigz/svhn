#!/usr/bin/env python
# coding: utf-8

# In[ ]:



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
from sklearn.metrics import accuracy_score
from keras import backend as K
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import cv2

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
tf.logging.set_verbosity(tf.logging.ERROR)

#extract tar fuction
def extract_tarball(filename, force=False):
    """ Helper function for extracting tarball files
    """
    # Drop the file extension
    root = filename.split('.')[0]

    # If file is already extracted - return
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
        return

    # If file is a tarball file - extract it
    if (filename.endswith("tar.gz")):
        print("Extracting %s ..." % filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()

#Grayscale function
def rgb2gray(image):
        """Convert images from rbg to grayscale
        """
        return np.expand_dims(np.dot(image, [0.2989, 0.5870, 0.1140]), axis=3)

def main():
    pass

def test(img):


    img_read = cv2.imread(img, -1)

    if len(img_read.shape) > 2 and img_read.shape[2] == 4:
        #convert the image from RGBA2RGB
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGRA2BGR)
    img_read = cv2.resize(img_read, (32, 32)).astype(np.float32)
    img_test = []

    img_test.append(np.asarray(img_read))
    img_test = np.array(img_test)

    img_test_greyscale = rgb2gray(img_test).astype(np.float32)

    new_model = load_model('multi_model.h5')

    score = new_model.predict(img_test_greyscale)

    score = np.concatenate(score, axis=1)

    roundedScore = np.zeros(score.shape, dtype="int32")
    roundedScore[score > 0.5] = 1

    digits = np.argmax(roundedScore[0:1,:7])
    start = 7
    hop = 11
    end = 0
    number = []
    for d in range(digits):
    	end = start+hop
    	dn = np.argmax(roundedScore[0:1,start:end])
    	start = end
    	if dn < 10:
    		number.append(dn)
    	else:
    		number.append('_')

    street_num = ''.join(map(str, number))
    print('\nIs it: ',street_num)
    print('\nNumber of Digits: ',np.argmax(roundedScore[0:1,:7]))
    if(np.argmax(roundedScore[0:1,7:18]) < 10):
    	print('\nFirst Digit: ',np.argmax(roundedScore[0:1,7:18]))
    if(np.argmax(roundedScore[0:1,18:29]) < 10):
    	print('\nSecond Digit: ',np.argmax(roundedScore[0:1,18:29]))
    if(np.argmax(roundedScore[0:1,29:40]) < 10):
    	print('\nThird Digit: ',np.argmax(roundedScore[0:1,29:40]))
    if(np.argmax(roundedScore[0:1,40:51]) < 10):
    	print('\nFourth Digit: ',np.argmax(roundedScore[0:1,40:51]))
    if(np.argmax(roundedScore[0:1,51:62]) < 10):
    	print('\nFifth Digit: ',np.argmax(roundedScore[0:1,51:62]))

    return street_num


def traintest():
    # In[ ]:


    path = "data"
    if not os.path.exists(path):
        os.mkdir(path)

    print('Downloading files.....\n')
    if os.path.isfile("data/train.tar.gz"):
        print('Train tarfile already downloaded')
    else:
        urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train.tar.gz", "data/train.tar.gz")
    if os.path.isfile("data/test.tar.gz"):
        print('Test tarfile already downloaded')
    else:
        urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/test.tar.gz", "data/test.tar.gz")
    if os.path.isfile("data/extra.tar.gz"):
        print('Extra tarfile already downloaded\n')
    else:
        urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/extra.tar.gz", "data/extra.tar.gz")
    print('files Downloaded\n')


    # In[ ]:





    # In[ ]:


    # Get the directory listing for the dataset folder
    ls_data = [f for f in os.listdir(path) if 'tar.gz' in f]

    # cd data
    os.chdir(path)

    # Extract the tarballs
    print('Extracting all tarfiles................\n')
    extract_tarball(ls_data[0])
    extract_tarball(ls_data[1])
    extract_tarball(ls_data[2])

    # cd ..
    os.chdir(os.path.pardir)


    # In[ ]:


    train_dstruct_path = "train/digitStruct.mat"
    extra_dstruct_path = "extra/digitStruct.mat"
    test_dstruct_path = "test/digitStruct.mat"


    # In[ ]:


    path = "data/"


    # In[ ]:

    print("\nAccessing Digitstruct files for data. This section Takes up to 3+ hours\n")


    print('Training set Digitstruct file...............\n')
    print("Read Digistsruct file to get boundarybox info and image Label data\n")

    if os.path.isfile("data/train_bbox_data.npy") and os.path.isfile("data/train_labels.npy"):
    	print("Train Preprocessed data already exists....skiping\n")
    else:


	    # In[ ]:train_labels


	    os.path.join(path, train_dstruct_path)


	    # In[ ]:


	    digit_struct_path = os.path.join(path, train_dstruct_path)
	    file = h5py.File(digit_struct_path, 'r')
	    size = len(file['digitStruct']['bbox'])


	    # In[ ]:


	    #tool to navigate the digitStruct file
	    #idx 1 two digits
	    #idx 34 one digit
	    idx = 34
	    item = file['digitStruct']['bbox'][idx].item()
	    attr = 'top'
	    num = len(file[item][attr])
	    value = []
	    if num == 1:
	        loc = file[item][attr]
	        value.append(loc.value[0][0])
	    else:
	        for i in range(num):
	            loc = file[item][attr].value[i].item()
	            value.append(file[loc].value[0][0])


	    # In[ ]:


	    height = []
	    width = []
	    left = []
	    top = []
	    label = []
	    imgsrc = []
	    attrs = ['height', 'label', 'left', 'top', 'width']
	    dataArrays = [height, label, left, top, width]
	    numDigits = np.zeros(size, dtype = int)

	    for idx in range(size):
	        item = file['digitStruct']['bbox'][idx].item()
	        #just read num of digits in 1 image, attr not important here
	        attr = 'top'
	        num = len(file[item][attr])
	        numDigits[idx] = num

	        for i in range(num):
	            imgsrc.append(idx+1)

	        for attrIdx in range(5):
	            attr = attrs[attrIdx]
	            dataArray = dataArrays[attrIdx]

	            if num == 1:
	                loc = file[item][attr]
	                dataArray.append(loc.value[0][0])
	            else:
	                value
	                for i in range(num):
	                    loc = file[item][attr].value[i].item()
	                    dataArray.append(file[loc].value[0][0])


	    # In[ ]:


	    print('Changing image label for number 0 from 10 to 0\n')
	    for n, i in enumerate(label):
	        if i == 10:
	            label[n] = 0


	    # In[ ]:


	    #save as npy
	    print("save boundarybox data for later use\n")

	    data = np.zeros((5,len(height)))
	    data[0] = np.array(height)
	    data[1] = np.array(width)
	    data[2] = np.array(left)
	    data[3] = np.array(top)
	    data[4] = np.array(imgsrc)
	    np.save('data/train_bbox_data.npy', data)


	    # In[ ]:


	    #convert label format to model output and save them separately

	    #first make 2d array for labels, each has at most 5 digits (digit 10 means invalid)
	    labelArray = np.zeros((size, int(np.max(numDigits))), dtype=int) + 10
	    digitIdx = 0
	    for i in range(1,size+1):
	        count = 0
	        while (digitIdx < len(label)) and (int(imgsrc[digitIdx]) == i):
	            labelArray[i-1,count] = label[digitIdx]
	            digitIdx += 1
	            count += 1


	    # In[ ]:


	    #represent no digit as 10 (categorical), possible value of digit(11)
	    #0-5 and more than 5, possible value of length(7) 7+11*5
	    values = np.zeros((size, 7+11*5), dtype=int)
	    values[:,:7] = to_categorical(np.minimum(numDigits,6), 7)
	    values[:,7:18] = to_categorical(labelArray[:,0], 11)
	    values[:,18:29] = to_categorical(labelArray[:,1], 11)
	    values[:,29:40] = to_categorical(labelArray[:,2], 11)
	    values[:,40:51] = to_categorical(labelArray[:,3], 11)
	    values[:,51:62] = to_categorical(labelArray[:,4], 11)

	    print('Save label data for later use\n')


	    np.save('data/train_labels.npy', values)


    # In[ ]:


    print("Test set Digitstruct files........\n")
    print("Read Digistsruct file to get boundarybox info and image Label data\n")
    if os.path.isfile("data/test_bbox_data.npy") and os.path.isfile("data/test_labels.npy"):
    	print("Test Preprocessed data already exists....skiping\n")
    else:


	    # In[ ]:


	    digit_struct_path = os.path.join(path, test_dstruct_path)
	    file = h5py.File(digit_struct_path, 'r')
	    size = len(file['digitStruct']['bbox'])


	    # In[ ]:


	    #tool to navigate the digitStruct file
	    idx = 34
	    item = file['digitStruct']['bbox'][idx].item()
	    #list(file[item].keys())
	    #['height', 'label', 'left', 'top', 'width']
	    attr = 'top'
	    num = len(file[item][attr])
	    value = []
	    #file[file[item]['height'].value[0].item()].value
	    if num == 1:
	        loc = file[item][attr]
	        value.append(loc.value[0][0])
	    else:
	        for i in range(num):
	            loc = file[item][attr].value[i].item()
	            value.append(file[loc].value[0][0])


	    # In[ ]:


	    height = []
	    width = []
	    left = []
	    top = []
	    label = []
	    imgsrc = []
	    attrs = ['height', 'label', 'left', 'top', 'width']
	    dataArrays = [height, label, left, top, width]
	    numDigits = np.zeros(size, dtype = int)

	    for idx in range(size):
	        item = file['digitStruct']['bbox'][idx].item()
	        #just read num of digits in 1 image, attr not important here
	        attr = 'top'
	        num = len(file[item][attr])
	        numDigits[idx] = num

	        for i in range(num):
	            imgsrc.append(idx+1)

	        for attrIdx in range(5):
	            attr = attrs[attrIdx]
	            dataArray = dataArrays[attrIdx]

	            if num == 1:
	                loc = file[item][attr]
	                dataArray.append(loc.value[0][0])
	            else:
	                value
	                for i in range(num):
	                    loc = file[item][attr].value[i].item()
	                    dataArray.append(file[loc].value[0][0])


	    # In[ ]:


	    print('Changing image label for number 0 from 10 to 0.............\n')
	    for n, i in enumerate(label):
	        if i == 10:
	            label[n] = 0


	    # In[ ]:


	    #save as npy
	    data = np.zeros((5,len(height)))
	    data[0] = np.array(height)
	    data[1] = np.array(width)
	    data[2] = np.array(left)
	    data[3] = np.array(top)
	    data[4] = np.array(imgsrc)

	    print("save boundarybox data for later use\n")
	    np.save('data/test_bbox_data.npy', data)


	    # In[ ]:


	    #convert label format to model output and save them separately

	    #first make 2d array for labels, each has at most 5 digits (digit 10 means invalid)
	    labelArray = np.zeros((size, int(np.max(numDigits))), dtype=int) + 10
	    digitIdx = 0
	    for i in range(1,size+1):
	        count = 0
	        while (digitIdx < len(label)) and (int(imgsrc[digitIdx]) == i):
	            labelArray[i-1,count] = label[digitIdx]
	            digitIdx += 1
	            count += 1


	    # In[ ]:


	    #represent no digit as 10 (categorical), possible value of digit(11)
	    #0-5 and more than 5, possible value of length(7) 7+11*5
	    values = np.zeros((size, 7+11*5), dtype=int)
	    values[:,:7] = to_categorical(np.minimum(numDigits,6), 7)
	    values[:,7:18] = to_categorical(labelArray[:,0], 11)
	    values[:,18:29] = to_categorical(labelArray[:,1], 11)
	    values[:,29:40] = to_categorical(labelArray[:,2], 11)
	    values[:,40:51] = to_categorical(labelArray[:,3], 11)
	    values[:,51:62] = to_categorical(labelArray[:,4], 11)

	    print("save label data for later use\n")


	    np.save('data/test_labels.npy', values)


    # In[ ]:


    print("Extra set Digitstruct files (Brace yourself)................\n")
    if os.path.isfile("data/extra_bbox_data.npy") and os.path.isfile("data/extra_labels.npy"):
    	print("Extra Preprocessed data already exists....skiping\n")
    else:


	    # In[ ]:


	    digit_struct_path = os.path.join(path, extra_dstruct_path)
	    file = h5py.File(digit_struct_path, 'r')
	    size = len(file['digitStruct']['bbox'])


	    # In[ ]:


	    #tool to navigate the digitStruct file
	    idx = 34
	    item = file['digitStruct']['bbox'][idx].item()
	    #list(file[item].keys())
	    #['height', 'label', 'left', 'top', 'width']
	    attr = 'top'
	    num = len(file[item][attr])
	    value = []
	    #file[file[item]['height'].value[0].item()].value
	    if num == 1:
	        loc = file[item][attr]
	        value.append(loc.value[0][0])
	    else:
	        for i in range(num):
	            loc = file[item][attr].value[i].item()
	            value.append(file[loc].value[0][0])


	    # In[ ]:


	    height = []
	    width = []
	    left = []
	    top = []
	    label = []
	    imgsrc = []
	    attrs = ['height', 'label', 'left', 'top', 'width']
	    dataArrays = [height, label, left, top, width]
	    numDigits = np.zeros(size, dtype = int)

	    for idx in range(size):
	        item = file['digitStruct']['bbox'][idx].item()
	        #just read num of digits in 1 image, attr not important here
	        attr = 'top'
	        num = len(file[item][attr])
	        numDigits[idx] = num

	        for i in range(num):
	            imgsrc.append(idx+1)

	        for attrIdx in range(5):
	            attr = attrs[attrIdx]
	            dataArray = dataArrays[attrIdx]

	            if num == 1:
	                loc = file[item][attr]
	                dataArray.append(loc.value[0][0])
	            else:
	                value
	                for i in range(num):
	                    loc = file[item][attr].value[i].item()
	                    dataArray.append(file[loc].value[0][0])



	    # In[ ]:


	    print('Changing image label for number 0 from 10 to 0.................\n')
	    for n, i in enumerate(label):
	        if i == 10:
	            label[n] = 0


	    # In[ ]:


	    #save as npy
	    data = np.zeros((5,len(height)))
	    data[0] = np.array(height)
	    data[1] = np.array(width)
	    data[2] = np.array(left)
	    data[3] = np.array(top)
	    data[4] = np.array(imgsrc)
	    np.save('data/extra_bbox_data.npy', data)


	    # In[ ]:


	    #convert label format to model output and save them separately

	    #first make 2d array for labels, each has at most 5 digits (digit 10 means invalid)
	    labelArray = np.zeros((size, int(np.max(numDigits))), dtype=int) + 10
	    digitIdx = 0
	    for i in range(1,size+1):
	        count = 0
	        while (digitIdx < len(label)) and (int(imgsrc[digitIdx]) == i):
	            labelArray[i-1,count] = label[digitIdx]
	            digitIdx += 1
	            count += 1


	    # In[ ]:


	    #represent no digit as 10 (categorical), possible value of digit(11)
	    #0-5 and more than 5, possible value of length(7) 7+11*5
	    values = np.zeros((size, 7+11*5), dtype=int)
	    values[:,:7] = to_categorical(np.minimum(numDigits,6), 7)
	    values[:,7:18] = to_categorical(labelArray[:,0], 11)
	    values[:,18:29] = to_categorical(labelArray[:,1], 11)
	    values[:,29:40] = to_categorical(labelArray[:,2], 11)
	    values[:,40:51] = to_categorical(labelArray[:,3], 11)
	    values[:,51:62] = to_categorical(labelArray[:,4], 11)


	    np.save('data/extra_labels.npy', values)


    # In[ ]:


    print(".......................\n")
    print("Crop Images and Convert to grayscale\n")
    print("Training Set Images\n")


    # In[ ]:


    train_data = np.load('data/train_bbox_data.npy')
    #reference: digitstruct_to_npy
    height = train_data[0]
    width = train_data[1]
    left = train_data[2]
    top = train_data[3]
    imgsrc = train_data[4]
    numdigits = train_data.shape[1]
    numImages = int(np.max(imgsrc))
    imgfolder = 'data/train/'


    # In[ ]:


    print("Crop Images at Box Boundary\n")
    savefolder = 'data/cropped_train_32/'
    if os.path.isdir(savefolder):
    	print("Cropped images already exist \n")
    else:
	    os.mkdir( savefolder );
	    size = 32, 32
	    #store all bounding box params(left,upper,right,lower) for all image
	    bbox = np.zeros((numImages,4))

	    i = 0


	    for imgNum in range(1,numImages+1):
	        num = 1
	        leftbound = left[i]
	        upperbound = top[i]
	        rightbound = left[i] + width[i]
	        lowerbound = top[i] + height[i]
	        i += 1

	        while (i < numdigits) and (int(imgsrc[i]) == imgNum):
	            leftbound = np.minimum(left[i],leftbound)
	            upperbound = np.minimum(top[i],upperbound)
	            rightbound = np.maximum(left[i] + width[i],rightbound)
	            lowerbound = np.maximum(top[i] + height[i],lowerbound)
	            num += 1
	            i += 1
	        #store bounding box params
	        bbox[imgNum-1] = np.array([leftbound,upperbound,rightbound,lowerbound])

	        img = Image.open(imgfolder+str(imgNum)+'.png').copy()
	        img = img.crop((leftbound, upperbound, rightbound, lowerbound))
	        img = img.resize(size)
	        img.save(savefolder+str(imgNum)+'.png')


    # In[ ]:


    print("Test Set Images\n")


    # In[ ]:


    test_data = np.load('data/test_bbox_data.npy')
    #reference: digitstruct_to_npy
    height = test_data[0]
    width = test_data[1]
    left = test_data[2]
    top = test_data[3]
    imgsrc = test_data[4]
    numdigits = test_data.shape[1]
    numImages = int(np.max(imgsrc))
    imgfolder = 'data/test/'


    # In[ ]:


    print("Crop Images at box boundary\n")
    ############ create the path to savefolder first#################
    #directly crop bbox containing all digits in 1 image to produce data
    savefolder = 'data/cropped_test_32/'
    if os.path.isdir(savefolder):
    	print("Cropped images already exist \n")
    else:
	    os.mkdir( savefolder );
	    size = 32, 32
	    #store all bounding box params(left,upper,right,lower) for all image
	    bbox = np.zeros((numImages,4))

	    i = 0


	    for imgNum in range(1,numImages+1):
	        num = 1
	        leftbound = left[i]
	        upperbound = top[i]
	        rightbound = left[i] + width[i]
	        lowerbound = top[i] + height[i]
	        i += 1

	        while (i < numdigits) and (int(imgsrc[i]) == imgNum):
	            leftbound = np.minimum(left[i],leftbound)
	            upperbound = np.minimum(top[i],upperbound)
	            rightbound = np.maximum(left[i] + width[i],rightbound)
	            lowerbound = np.maximum(top[i] + height[i],lowerbound)
	            num += 1
	            i += 1
	        #store bounding box params
	        bbox[imgNum-1] = np.array([leftbound,upperbound,rightbound,lowerbound])

	        img = Image.open(imgfolder+str(imgNum)+'.png').copy()
	        img = img.crop((leftbound, upperbound, rightbound, lowerbound))
	        img = img.resize(size)
	        img.save(savefolder+str(imgNum)+'.png')


    # In[ ]:


    print("Extra Set Images\n")
    extra_data = np.load('data/extra_bbox_data.npy')
    height = extra_data[0]
    width = extra_data[1]
    left = extra_data[2]
    top = extra_data[3]
    imgsrc = extra_data[4]
    numdigits = extra_data.shape[1]
    numImages = int(np.max(imgsrc))
    imgfolder = 'data/extra/'


    # In[ ]:


    print("Crop Each image in Extra set at box boundary border 1 by 1 (I am sorry) :) \n")
    savefolder = 'data/cropped_extra_32/'
    if os.path.isdir(savefolder):
    	print("Cropped images already exist \n")
    else:
	    os.mkdir( savefolder );
	    size = 32, 32
	    #store all bounding box params(left,upper,right,lower) for all image
	    bbox = np.zeros((numImages,4))

	    i = 0
	    for imgNum in range(1,numImages+1):
	        num = 1
	        leftbound = left[i]
	        upperbound = top[i]
	        rightbound = left[i] + width[i]
	        lowerbound = top[i] + height[i]
	        i += 1

	        while (i < numdigits) and (int(imgsrc[i]) == imgNum):
	            leftbound = np.minimum(left[i],leftbound)
	            upperbound = np.minimum(top[i],upperbound)
	            rightbound = np.maximum(left[i] + width[i],rightbound)
	            lowerbound = np.maximum(top[i] + height[i],lowerbound)
	            num += 1
	            i += 1
	        #store bounding box params
	        bbox[imgNum-1] = np.array([leftbound,upperbound,rightbound,lowerbound])

	        img = Image.open(imgfolder+str(imgNum)+'.png').copy()
	        img = img.crop((leftbound, upperbound, rightbound, lowerbound))
	        img = img.resize(size)
	        img.save(savefolder+str(imgNum)+'.png')


    # In[ ]:


    train_labels = np.load('data/train_labels.npy')


    # In[ ]:


    print("Store cropped training images into an array\n")
    size = train_labels.shape[0]
    folder = 'data/cropped_train_32/'
    train_images = []


    for i in range(size):
        im = Image.open(folder+str(i+1)+'.png')
        train_images.append(np.asarray(im))

    train_images = np.array(train_images)


    # In[ ]:

    test_labels = np.load('data/test_labels.npy')


    # In[ ]:


    print("Store cropped test images into an array\n")
    size = test_labels.shape[0]
    folder = 'data/cropped_test_32/'
    test_images = []


    for i in range(size):
        im = Image.open(folder+str(i+1)+'.png')
        test_images.append(np.asarray(im))


    test_images = np.array(test_images)


    # In[ ]:


    extra_labels = np.load('data/extra_labels.npy')
    print("Store cropped Extra images into an array one by one :D \n")
    size = extra_labels.shape[0]
    folder = 'data/cropped_extra_32/'
    extra_images = []

    for i in range(size):
        im = Image.open(folder+str(i+1)+'.png')
        extra_images.append(np.asarray(im))


    extra_images = np.array(extra_images)


    # In[ ]:


    X_train = train_images.copy()
    X_test = test_images.copy()
    X_extra = extra_images.copy()

    print("convert each image set to grayscale\n")

    train_greyscale = rgb2gray(X_train).astype(np.float32)
    test_greyscale = rgb2gray(X_test).astype(np.float32)
    extra_greyscale = rgb2gray(X_extra).astype(np.float32)

    print("Dimensions")
    print("Training set", train_greyscale.shape, "\tLabels", train_labels.shape)
    print("Test set", test_greyscale.shape, "\tLabels", test_labels.shape)
    print("Extra set", extra_greyscale.shape, "\tLabels", extra_labels.shape)
    print('')

    print("Normalize all images to be within 0 and 1 by dividing by 255...\n")

    train_greyscale /= 255
    test_greyscale /= 255
    extra_greyscale /= 255


    # In[ ]:


    print('Randomly collect 20% from both training and extra dataset each and combine to create a validation set\n')
    print('Combine remaining 80% in training and extra to make full training set for model\n')

    X_test = test_greyscale.copy()
    y_test = test_labels.copy()
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(train_greyscale, train_labels, test_size=0.20, shuffle= True, random_state=66)
    X_extra_train, X_extra_val, y_extra_train, y_extra_val = train_test_split(extra_greyscale, extra_labels, test_size=0.20, shuffle= True, random_state=66)

    X_train = np.concatenate([X_train_train, X_extra_train])
    y_train = np.concatenate([y_train_train, y_extra_train])

    X_val = np.concatenate([X_train_val, X_extra_val])
    y_val = np.concatenate([y_train_val, y_extra_val])

    print('Full Training set', X_train.shape, y_train.shape)
    print('Full Validation set', X_val.shape, y_val.shape)


    # In[ ]:


    print('Build Model...............\n')


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
    o_length = Dense(7, activation='softmax')(d2d)
    o_digit1 = Dense(11, activation='softmax')(d2d)
    o_digit2 = Dense(11, activation='softmax')(d2d)
    o_digit3 = Dense(11, activation='softmax')(d2d)
    o_digit4 = Dense(11, activation='softmax')(d2d)
    o_digit5 = Dense(11, activation='softmax')(d2d)
    model = Model(a, [o_length,o_digit1,o_digit2,o_digit3,o_digit4,o_digit5])

    model.compile(Adam(lr=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    # In[ ]:


    print('Fit Model..........................\n')


    # In[ ]:


    history = model.fit(X_train, [y_train[:,:7], y_train[:,7:18], y_train[:,18:29], y_train[:,29:40], y_train[:,40:51], y_train[:,51:62]], validation_data=(X_val, [y_val[:,:7], y_val[:,7:18], y_val[:,18:29], y_val[:,29:40], y_val[:,40:51], y_val[:,51:62]]), epochs=100, batch_size=1000)


    # In[ ]:


    print('Test Model\n')
    #validation
    score = model.predict(X_test)

    score = np.concatenate(score, axis=1)
    #round score to get actual prediction instead of probability
    roundedScore = np.zeros(score.shape, dtype="int32")
    #choose threshold
    roundedScore[score > 0.5] = 1

    #rounded acc (exactly match)
    print("Test Set Accuracy: ",accuracy_score(y_test, roundedScore))

    print('F1 scores\n')

    print("\nFor Digit length of Numbers:", f1_score(y_test[:,:7].argmax(axis=1), roundedScore[:,0:7].argmax(axis=1), average='macro'))
    print(classification_report(y_test[:,:7].argmax(axis=1), roundedScore[:,0:7].argmax(axis=1)))

    print("\nFor First Numbers: ",f1_score(y_test[:,7:18].argmax(axis=1), roundedScore[:,7:18].argmax(axis=1), average='macro'))
    print(classification_report(y_test[:,7:18].argmax(axis=1), roundedScore[:,7:18].argmax(axis=1)))

    print("\nFor Second Numbers: ",f1_score(y_test[:,18:29].argmax(axis=1), roundedScore[:,18:29].argmax(axis=1), average='macro'))
    print(classification_report(y_test[:,18:29].argmax(axis=1), roundedScore[:,18:29].argmax(axis=1)))

    print("\nFor Third Numbers: ",f1_score(y_test[:,29:40].argmax(axis=1), roundedScore[:,29:40].argmax(axis=1), average='macro'))
    print(classification_report(y_test[:,29:40].argmax(axis=1), roundedScore[:,29:40].argmax(axis=1)))

    print("\nFor Fourth Numbers: ",f1_score(y_test[:,40:51].argmax(axis=1), roundedScore[:,40:51].argmax(axis=1), average='macro'))
    print(classification_report(y_test[:,40:51].argmax(axis=1), roundedScore[:,40:51].argmax(axis=1)))

    print("\nFor Fifth Numbers: ",f1_score(y_test[:,51:62].argmax(axis=1), roundedScore[:,51:62].argmax(axis=1), average='macro'))
    print(classification_report(y_test[:,51:62].argmax(axis=1), roundedScore[:,51:62].argmax(axis=1)))

    finished = 'finished'
    return finished

