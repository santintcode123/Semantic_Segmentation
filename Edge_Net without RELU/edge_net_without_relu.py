# -*- coding: utf-8 -*-

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

dict_data=load('/content/drive/My Drive/Colab Notebooks/data.npz')

X_train = dict_data['arr_0']
Y_train = dict_data['arr_1']
edgemap = dict_data['arr_2']
X_val = dict_data['arr_3']
Y_val = dict_data['arr_4']
X_test = dict_data['arr_5']
print(X_train.shape)
print(Y_train.shape)
print(edgemap.shape)
train_length = 28280
val_length = 5000
test_length = 5000

image_x = random.randint(0, train_length)
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
imshow(np.squeeze(edgemap[image_x]))
plt.show()

def EdgeNet():
  inputs=tf.keras.layers.Input((128,128,1))
  e1 = tf.keras.layers.Conv2D(16, (3,3), padding="same" )( inputs )
  e2 = tf.keras.layers.Conv2D(32, (3,3), padding="same" )( e1 )
  outputs = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid" )( e2 )

  edge_model=tf.keras.models.Model(inputs,outputs)
  return edge_model

edge_model=EdgeNet()
edge_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
edge_model.summary()

results = edge_model.fit(Y_train, edgemap ,validation_split=0.1, batch_size=32, epochs=30)
edge_model.save('Edge_Net_without_RELU.h5')

preds_edge_train = edge_model.predict( Y_train, verbose=1 )
preds_edge_val = edge_model.predict( Y_val, verbose=1 )

preds_edge_train_t = (preds_edge_train > 0.5).astype( np.uint8 )
preds_edge_val_t = (preds_edge_val > 0.5).astype( np.uint8 )

ix1 = random.randint(0, train_length)

imshow(X_train[ix1])
plt.show()
imshow(np.squeeze(Y_train[ix1]))
plt.show()
imshow(np.squeeze(preds_edge_train_t[ix1]))
plt.show()

ix3 = random.randint(0, val_length)
imshow(X_val[ix3])
plt.show()
imshow(np.squeeze(Y_val[ix3]))
plt.show()
imshow(np.squeeze(preds_edge_val_t[ix3]))
plt.show()