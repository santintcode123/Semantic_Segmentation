# -*- coding: utf-8 -*-

zip_path = '/content/drive/My Drive/Colab Notebooks/x_train.zip'
!cp "{zip_path}" . 
!unzip -q x_train.zip
!rm x_train.zip

zip_path = '/content/drive/My Drive/Colab Notebooks/y_train.zip'
!cp "{zip_path}" . 
!unzip -q y_train.zip
!rm y_train.zip

zip_path = '/content/drive/My Drive/Colab Notebooks/x_test.zip'
!cp "{zip_path}" . 
!unzip -q x_test.zip
!rm x_test.zip

zip_path = '/content/drive/My Drive/Colab Notebooks/x_val.zip'
!cp "{zip_path}" . 
!unzip -q x_val.zip
!rm x_val.zip

zip_path = '/content/drive/My Drive/Colab Notebooks/y_val.zip'
!cp "{zip_path}" . 
!unzip -q y_val.zip
!rm y_val.zip

import tensorflow as tf
import os
import random
import numpy as np
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import  gray2rgb, rgb2gray
import matplotlib.pyplot as plt

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

X_TRAIN_PATH = '/content/x_train/'
Y_TRAIN_PATH = '/content/y_train/'
X_TEST_PATH = '/content/x_test/'
X_VAL_PATH = '/content/x_val/'
Y_VAL_PATH = '/content/y_val/'

seed=42
random.seed = seed
np.random.seed = seed

import os 
train_ids = next( os.walk( X_TRAIN_PATH ) )[2]
test_ids = next( os.walk( X_TEST_PATH ) )[2]
val_ids = next( os.walk( X_VAL_PATH ) )[2]
print( "Length of Trains ids" , len( train_ids ) )
print( "Length of Test ids  ", len( test_ids ) )
print( "Length of Val ids  ", len( val_ids ) )

X_train = np.zeros( (len( train_ids ), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8 )
Y_train = np.zeros( (len( train_ids ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
X_val= np.zeros( (len( val_ids ), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8 )
Y_val= np.zeros( (len( val_ids ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
synt = np.zeros ( (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8 )
edgemap = np.zeros( (len( train_ids ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
c=0

print( 'Resizing training images' )
for n, id_ in tqdm( enumerate( train_ids ), total=len( train_ids ) ):
    img = imread( X_TRAIN_PATH + id_ )
    img = resize( img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True )
    if img.shape != synt.shape :
        img = gray2rgb(img)
    X_train[n] = img
    img2 = imread( Y_TRAIN_PATH + id_[:7] + '.png')
    img2 = resize( img2, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True )
    for i in range(128):
        for j in range(128):
            for k in range(3):
                if img2[i,j,k]>0 :
                    img2[i,j,0]=255
                    img2[i,j,1]=255
                    img2[i,j,2]=255
    img2=rgb2gray(img2)
    for i in range(1, 127):
      for j in range(1, 127):
        if ((img2[i-1,j-1] > 128) or (img2[i-1,j] > 128) or (img2[i-1,j+1] > 128) or (img2[i,j-1] > 128) or (img2[i,j+1] > 128) or (img2[i+1,j-1] > 128) or (img2[i+1,j] > 128) or (img2[i+1,j+1] > 128) ):
          if ((img2[i-1,j-1] > 128) and (img2[i-1,j] > 128) and (img2[i-1,j+1] > 128) and (img2[i,j-1] > 128) and (img2[i,j+1] > 128) and (img2[i+1,j-1] > 128) and (img2[i+1,j] > 128) and (img2[i+1,j+1] > 128) ):
             edgemap[c,i,j,0]=0
          else : 
             edgemap[c,i,j,0]=255
    c=c+1
    img2 = img2[:, :, np.newaxis]
    Y_train[n] = img2

print( 'Resizing Validation images' )
for n, id_ in tqdm( enumerate( val_ids ), total=len( val_ids ) ):
    img = imread( X_VAL_PATH + id_ )
    img = resize( img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True )
    if img.shape != synt.shape :
        img = gray2rgb(img)   
    X_val[n] = img
    img2 = imread( Y_VAL_PATH + id_[:7] + '.png')
    img2 = resize( img2, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True )
    for i in range(128):
        for j in range(128):
            for k in range(3):
                if img2[i,j,k]>0 :
                    img2[i,j,0]=255
                    img2[i,j,1]=255
                    img2[i,j,2]=255
    img2=rgb2gray(img2)
    img2 = img2[:, :, np.newaxis]
    Y_val[n] = img2

X_test = np.zeros( (len( test_ids ), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8 )
print( 'Resizing test images' )
for n, id_ in tqdm( enumerate( test_ids ), total=len( test_ids ) ):
    img3 = imread( X_TEST_PATH + id_ )
    img3 = resize( img3, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True )
    if img3.shape != synt.shape :
        img3 = img3[:, :, np.newaxis]
    X_test[n] = img3


print( 'Data Pre-processing Done' )

data = asarray(X_train)
data1 = asarray(Y_train)
data2 = asarray(edgemap)
data3 = asarray(X_val)
data4 = asarray(Y_val)
data5= asarray(X_test)

savez_compressed('data.npz', data,data1,data2,data3,data4,data5)

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
imshow(np.squeeze(edgemap[image_x]))
plt.show()