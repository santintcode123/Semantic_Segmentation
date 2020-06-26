

import tensorflow as tf
import os
import random
import numpy as np
import keras.backend as K
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from tqdm import tqdm

from tensorflow.keras.callbacks import TensorBoard
from skimage.io import imread, imshow , imsave
from skimage.transform import resize
from skimage.color import  gray2rgb, rgb2gray
import matplotlib.pyplot as plt

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

dict_data=load('data.npz')

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

model1 = tf.keras.models.load_model('Edge_Net_without_RELU.h5')
model1.summary()

model1.layers[0]._name='zero'
model1.layers[1]._name='first'
model1.layers[2]._name='second'
model1.layers[3]._name='third'

lambda1=1
lambda2=0.5
lambda3=0.25

def loss_fn(y_true,y_predict):
  y_true=tf.cast(y_true,tf.float32)
  y_predict=tf.cast(y_predict,tf.float32)
  
  
  # for layers 1  (128,128,16)
  layer_output1=model1.get_layer('first').output
  
  model_layer1=tf.keras.models.Model(inputs=model1.input,outputs=layer_output1)
  
  E1_hat=model_layer1(y_predict)
  E1=model_layer1(y_true)
  
  diff1=tf.keras.backend.abs(E1_hat-E1)
  
  # for layers 2 (128,128,32)
  layer_output2=model1.get_layer('second').output
  model_layer2=tf.keras.models.Model(inputs=model1.input,outputs=layer_output2)
  E2_hat=model_layer2(y_predict)
  E2=model_layer2(y_true)
  diff2=tf.keras.backend.abs(E2_hat-E2)
  

  #for output layer (128,128,1)
  E3_hat=model1(y_predict)
  E3=model1(y_true)
  diff3=tf.keras.backend.abs(E3_hat-E3)
  
  
  #total loss
  sum1=(lambda1*tf.reduce_sum(diff1))  + (lambda2*tf.reduce_sum(diff2))+(lambda3*tf.reduce_sum(diff3))
  
  
  sum2=tf.keras.losses.binary_crossentropy(y_true,y_predict)
  sum2=sum2+sum1

  return sum2

model2 = tf.keras.models.load_model('model_mobileNet_with_semeda_withoutRELU.h5', custom_objects={'loss_fn' : loss_fn})
model2.summary()

# import the opencv library
import cv2

# define a video capture object
vid = cv2.VideoCapture( 0 )

while (True):

  # Capture the video frame
  # by frame
  ret, frame = vid.read()
  # Display the resulting frame
  cv2.imshow( 'frame', frame )

  img = resize( frame , (128, 128), mode='constant', preserve_range=True )
  img = tf.cast( img, dtype=np.uint8 )
  img = tf.reshape( img, (1, 128, 128, 3) )
#  plt.imshow( img[0] )

  yhat = model2.predict( img )
  #yhat_t = (yhat > 0.5).astype( np.uint8 )
  yhat = gray2rgb( yhat )  # Comment this line for Gray Scale Image
  disp = np.squeeze( yhat[0] )
  cv2.imshow( 'disp', disp )

  # the 'q' button is set as the
  # quitting button you may use any
  # desired button of your choice
  if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
    break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


