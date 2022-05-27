## TODO:
- Write Content
- Publish

# digit_recog
Web app to recognize digits using Keras ML model, trained on the mnist dataset.

This web app deploys a wide variety of technologies, including Keras, Flask, Bootstrap, and html.

# Data

The Modified National Institute of Standards and Technology (MNIST) dataset of handwritten digits was used as the training and validation datasets for this model. The dataset was obtained from the keras.datasets module as follows:

```
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

This dataset consists of 60,000 training and 10,000 testing digits and their corresponding labels.

The data is normalized to have a mean value of 0 and standard deviaiton of 1.

Data augmentation is performed the prevent overfitting of the dataset, using the following  random operations:
- 10 degree rotations
- 10% zoom
- 10% horizontal shifts
- 10% vertical shifts


# Model

The model used is the LeNet-5 v2.0 convolutional neural network (CNN), originally presented in [LeCun et al., Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).

The model was implemented in Keras, as shown here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/turnerluke/digit_recog/blob/main/models/LeNet_5_train.ipynb)

The structure of this CNN is as follows:

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 convolution_1 (Conv2D)      (None, 28, 28, 32)        832       
                                                                 
 convolution_2 (Conv2D)      (None, 24, 24, 32)        25600     
                                                                 
 batchnorm_1 (BatchNormaliz  (None, 24, 24, 32)        128       
 ation)                                                           
                                                                 
 activation_25 (Activation)  (None, 24, 24, 32)        0         
                                                                 
 max_pool_1 (MaxPooling2D)   (None, 12, 12, 32)        0         
                                                                 
 dropout_1 (Dropout)         (None, 12, 12, 32)        0         
                                                                 
 convolution_3 (Conv2D)      (None, 10, 10, 64)        18496     
                                                                 
 convolution_4 (Conv2D)      (None, 8, 8, 64)          36864     
                                                                 
 batchnorm_2 (BatchNormaliz  (None, 8, 8, 64)          256       
 ation)                                                           
                                                                 
 activation_26 (Activation)  (None, 8, 8, 64)          0         
                                                                 
 max_pool_2 (MaxPooling2D)   (None, 4, 4, 64)          0         
                                                                 
 dropout_2 (Dropout)         (None, 4, 4, 64)          0         
                                                                 
 flatten (Flatten)           (None, 1024)              0         
                                                                 
 fully_connected_1 (Dense)   (None, 256)               262144    
                                                                 
 batchnorm_3 (BatchNormaliz  (None, 256)               1024      
 ation)                                                           
                                                                 
 activation_27 (Activation)  (None, 256)               0         
                                                                 
 fully_connected_2 (Dense)   (None, 128)               32768     
                                                                 
 batchnorm_4 (BatchNormaliz  (None, 128)               512       
 ation)                                                           
                                                                 
 activation_28 (Activation)  (None, 128)               0         
                                                                 
 fully_connected_3 (Dense)   (None, 84)                10752     
                                                                 
 batchnorm_5 (BatchNormaliz  (None, 84)                336       
 ation)                                                           
                                                                 
 activation_29 (Activation)  (None, 84)                0         
                                                                 
 dropout_3 (Dropout)         (None, 84)                0         
                                                                 
 output (Dense)              (None, 10)                850       
                                                                 
=================================================================
```

<img src="/images/LeNet 5 v2 Vis.png">

**Figure 1:** *Model CNN Visualization.*

# Application

The web app presents a clean user interface containing a canvas to draw the digits, a clear button, a predict button, a descriptive prediciton, and some miscellaneous information about the project.

The website was constructed in [Flask](https://flask.palletsprojects.com/en/2.1.x/f).
