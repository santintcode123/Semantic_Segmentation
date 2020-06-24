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
Y_train = np.zeros( (len( train_ids ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool )
X_val= np.zeros( (len( val_ids ), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8 )
Y_val= np.zeros( (len( val_ids ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool )
synt = np.zeros ( (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8 )

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

image_x = random.randint(0, len(train_ids))
plt.imshow(X_train[image_x])
plt.show()
plt.imshow(np.squeeze(Y_train[image_x]))
plt.show()

def sepConv(x,num_filter,strides=1):
  x=tf.keras.layers.DepthwiseConv2D(kernel_size=3,padding='same')(x)
  x=tf.keras.layers.BatchNormalization()(x)
  x=tf.keras.layers.Activation('relu')(x)
  x=tf.keras.layers.Conv2D(num_filter,kernel_size=(1,1),padding='same',strides=strides)(x)
  x=tf.keras.layers.BatchNormalization()(x)
  x=tf.keras.layers.Activation('relu')(x)
  return x

def Conv(x,num_filter,kernel_size=3,strides=1):
  x=tf.keras.layers.Conv2D(num_filter,kernel_size=kernel_size,padding='same',strides=strides)(x)
  x=tf.keras.layers.BatchNormalization()(x)
  x=tf.keras.layers.Activation('relu')(x)
  return x

def MobNet():
  inputs=tf.keras.layers.Input((128,128,3))
  x1=Conv(inputs,num_filter=32,kernel_size=3,strides=1)
  x1=sepConv(x1,num_filter=32,strides=1)       # 128*128*32
  # down_sample block 2
  x2=tf.keras.layers.MaxPool2D((2,2),(2,2))(x1)
  x2=Conv(x2,num_filter=64,kernel_size=3)
  x2=sepConv(x2,num_filter=64,strides=1)   # 64*64*64
  # down_sample block 3
  x3=tf.keras.layers.MaxPool2D((2,2),(2,2))(x2)
  x3=Conv(x3,num_filter=128,kernel_size=3)
  x3=sepConv(x3,num_filter=128)   #32*32*128
  #down_sample block 4
  x4=tf.keras.layers.MaxPool2D((2,2),(2,2))(x3)
  x4=Conv(x4,num_filter=256,kernel_size=1)
  x4=sepConv(x4,num_filter=256)  #16*16*256

  #down_sample block 5
  x5=tf.keras.layers.MaxPool2D((2,2),(2,2))(x4)
  x5=Conv(x5,num_filter=512,kernel_size=1)
  x5=sepConv(x5,num_filter=512)    # 8*8*512
  # up_sampling block 1
  u1=tf.keras.layers.UpSampling2D((2,2))(x5)
  u1=tf.keras.layers.Concatenate()([u1,x4]) # 16*16*(512+256)
  u1=Conv(u1,num_filter=256,kernel_size=1)
  u1=sepConv(u1,num_filter=256) # 16*16*256
  #upsampling_block 2
  u2=tf.keras.layers.UpSampling2D((2,2))(u1)
  u2=tf.keras.layers.Concatenate()([u2,x3])  # 32*32*(256+128)
  u2=Conv(u2,num_filter=128,kernel_size=1)
  u2=sepConv(u2,num_filter=128)   # 32*32*128
  # upsampling block 3
  u3=tf.keras.layers.UpSampling2D((2,2))(u2)
  u3=tf.keras.layers.Concatenate()([u3,x2])
  u3=Conv(u3,num_filter=64,kernel_size=1)
  u3=sepConv(u3,num_filter=64) # 64*64*64
  # upsampling block 4
  u4=tf.keras.layers.UpSampling2D((2,2))(u3)
  u4=tf.keras.layers.Concatenate()([u4,x1])
  u4=Conv(u4,num_filter=32)
  u4=sepConv(u4,num_filter=32) 
  outputs=tf.keras.layers.Conv2D(1,(1,1),padding='same',activation='sigmoid')(u4)
  
  
  model=tf.keras.models.Model(inputs,outputs)
  return model

#def dice_loss2(y_true, y_pred):
 #   y_true = tf.cast( y_true, tf.float32 )
  #  y_pred = tf.cast( y_pred, tf.float32 )
   # numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    #denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    #return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

def myloss(y_true, y_pred):
    y_true = tf.cast( y_true, tf.float32 )
    y_pred = tf.cast( y_pred, tf.float32 )
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

model=MobNet()
model.compile(optimizer="adam", loss= myloss, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
model.summary()

tensorboard_callback = TensorBoard(log_dir='./logs')
results = model.fit(X_train, Y_train,validation_data=(X_val,Y_val), batch_size=16, epochs=15, callbacks=[tensorboard_callback])
val_loss,val_accuracy = model.evaluate(X_val, Y_val)
print(val_loss,val_accuracy*100)

model.save("model_mobileNetDICELOSS+crossentropy.h5")

preds_train = model.predict( X_train, verbose=1 )
preds_val = model.predict( X_val, verbose=1 )
preds_test = model.predict( X_test, verbose=1 )
preds_train_t = (preds_train > 0.5).astype( np.uint8 )
preds_val_t = (preds_val > 0.5).astype( np.uint8 )
preds_test_t = (preds_test > 0.5).astype( np.uint8 )

ix1 = random.randint(0, len(preds_train_t))
imshow(X_train[ix1])
plt.show()
imshow(np.squeeze(Y_train[ix1]))
plt.show()
imshow(np.squeeze(preds_train_t[ix1]))
plt.show()

ix2 = random.randint(0, len(preds_val_t))
imshow(X_val[ix2])
plt.show()
imshow(np.squeeze(Y_val[ix2]))
plt.show()
imshow(np.squeeze(preds_val_t[ix2]))
plt.show()

ix3 = random.randint(0, len(preds_test_t))
imshow(X_test[ix3])
plt.show()
imshow(np.squeeze(preds_test_t[ix3]))
plt.show()