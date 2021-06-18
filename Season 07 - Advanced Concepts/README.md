<img src="scenarios/media/saccaldic_motion.jpg" align="right" alt="" width="225"/>

# Advacnced Concepts 
- **Advance Convolutions**
- **Attention and Image Augmentation: Depthwise**
- **Pixel Shuffle** 
- **Dilated, Transpose**
- **Channel Attention**

**Objective** : To achieve 87% accuracy with total Params less than 100k in CIFAR10 dataset

## Data Analysis

Generally, when you have to deal with image, text, audio or video data, you can use standard python packages that load data into a numpy array. Then you can convert this array into a **`torch.*Tensor`**.

    For images, packages such as Pillow, OpenCV are useful
    For audio, packages such as scipy and librosa
    For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful

Specifically for vision, we have created a package called torchvision, that has data loaders for common datasets such as **Imagenet, CIFAR10, MNIST, etc**. and data transformers for images, viz., torchvision.datasets and torch.utils.data.DataLoader.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the **CIFAR10** dataset. It has the classes: **‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’**`. The images in **CIFAR-10** are of size **3x32x32**, i.e. 3-channel color images of 32x32 pixels in size.

## Model

	----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
	================================================================
            Conv2d-1           [-1, 10, 32, 32]             280
              ReLU-2           [-1, 10, 32, 32]               0
       BatchNorm2d-3           [-1, 10, 32, 32]              20
           Dropout-4           [-1, 10, 32, 32]               0
            Conv2d-5           [-1, 16, 30, 30]           4,016
              ReLU-6           [-1, 16, 30, 30]               0
       BatchNorm2d-7           [-1, 16, 30, 30]              32
           Dropout-8           [-1, 16, 30, 30]               0
         MaxPool2d-9           [-1, 16, 15, 15]               0
           Conv2d-10           [-1, 32, 15, 15]             544
           Conv2d-11           [-1, 32, 15, 15]           9,248
             ReLU-12           [-1, 32, 15, 15]               0
      BatchNorm2d-13           [-1, 32, 15, 15]              64
          Dropout-14           [-1, 32, 15, 15]               0
        MaxPool2d-15             [-1, 32, 7, 7]               0
           Conv2d-16             [-1, 64, 5, 5]          18,496
           Conv2d-17             [-1, 64, 5, 5]          36,928
             ReLU-18             [-1, 64, 5, 5]               0
      BatchNorm2d-19             [-1, 64, 5, 5]             128
          Dropout-20             [-1, 64, 5, 5]               0
           Conv2d-21             [-1, 64, 5, 5]           1,216
           Conv2d-22             [-1, 64, 5, 5]           4,160
             ReLU-23             [-1, 64, 5, 5]               0
      BatchNorm2d-24             [-1, 64, 5, 5]             128
          Dropout-25             [-1, 64, 5, 5]               0
        MaxPool2d-26             [-1, 64, 2, 2]               0
           Conv2d-27             [-1, 64, 2, 2]          36,928
             ReLU-28             [-1, 64, 2, 2]               0
      BatchNorm2d-29             [-1, 64, 2, 2]             128
          Dropout-30             [-1, 64, 2, 2]               0
	AdaptiveAvgPool2d-31             [-1, 64, 1, 1]               0
           Linear-32                   [-1, 10]             650
	================================================================
    
## Logs

