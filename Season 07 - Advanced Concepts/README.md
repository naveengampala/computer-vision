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
