# Real Time Semantic_Segmentation
The goal of Semantic Segmentation is to label each pixel of an image with a corresponding class of what is being represented. This github Repo is mainly focuses on human classes and differentiate human class from others and mask out only humans.An RGB color image (height.width.3) or a grayscale image (height.width.1) is given as input and output a segmentation map where each pixel is categorized as a human pixel or not.

# Description
The goal is to easily be able to implement, train, and test new Semantic Segmentation models! Complete with the following:

1. Training and testing 
2. Several state-of-the-art models. Easily plug and play with different models
3. CIHP dataset
4. Evaluation with mean IoU score
5. Plotting of loss function and accuracy over epochs using tensorboard callback

# Prerequisites
-Python
-Keras 
-Opencv for python
-Tensorflow 
-Numpy

# Where it can be used
- The tflite file attached can be used with Android Studio to build android apps based on semantic segmentation.
- Models can be trained using other data such as COCO dataset and PASCAL VOC datasets.

# How to Use
- CIHP data is preprocessed for semantic segmentation and stored in .npz file.
- The edge maps for ground truth masks are also stored in .npz file.
- Each model is pretrained and saved as .h5 file. These models can be used directly by loading them using keras model load API.
- Models can also be trained by changing hyperparamters using code provided for each model in model folder.
- To run demo, add model required and if it has any other dependent model, add it also.
- Change the path of data.npz
- Change the path of the models to the required path in finalmodel.py 
- Run the code and get the segmented masks of a live feed using your webcam or any other device.






# Contributors
B Sai Akshay
Santosh Kumar Meena







