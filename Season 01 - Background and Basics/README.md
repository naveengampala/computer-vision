# Background and Basics</br>

### What are `Channels` and `Kernels`?</br>
**`Channels`** come from `media`. Looking at broadcast technology behind TVs you have mulitple channels for different information that gets broadcasted to your TV. Let's assume that we are talking about 2D convolutions applied on images.</br>

In a grayscale image, the data is a matrix of dimensions **_w×h_**, where **_w_** is the `width` of the image and **_h_** is its `height`. In a color image, we normally have 3 channels: `red, green and blue` ; this way, a color image can be represented as matrix of dimensions **_w×h×c_**, where **_c_** is the `number of channels`, that is, 3.
    
A convolution layer receives the image **_w×h×c_** as input, and generates as output an activation map of dimensions **_w′×h′×c′_**. The number of input channels in the convolution is c, while the number of output channels is **_c′_** . The filter for such a convolution is a tensor of dimensions **_f×f×c×c′_** , where **_f_** is the filter size (normally 3 or 5).

This way, the number of channels is the `depth of the matrices involved in the convolutions`. Also, a convolution operation defines the variation in such depth by specifying input and output channels.

These explanations are directly extrapolable to 1D signals or 3D signals, but the analogy with image channels made it more appropriate to use 2D signals in the example.

<img src="../scenarios/media/kernels.jpg" align="right" alt="" width="225"/> </br>
**`Kernels`** are nothing but a filter that is used to extract the features from the images. The kernel is a matrix that moves over the input data, performs the dot product with the sub-region of input data, and gets the output as the matrix of dot products. Kernel moves on the input data by the stride value. If the stride value is 2, then kernel moves by 2 columns of pixels in the input matrix. In short, the kernel is used to extract high-level features like edges from the image.

### Why should we (nearly) _always use 3x3 kernels_?</br>
This question can be answered in two parts. `One part` is to answer `why noy even kernels` _(2x2, 4x4)_ and the `second part` is why not using `bigger kernels`_(5x5, 7x7, 9x9..)_ .

First, with even kernels the problem is its difficult to find axis of symmetry. Without centre point, it is difficult to depict information in a symmetric way.

Second, using a `higher size kernel increases the computation cost` with more number of parameters and also the amount of information or features extracted are considerably lesser (as the dimension of next layer reduces greatly). Using a `lower size kernel like 1x1 does not account of features from the neighbouring pixels`, 1x1 is used only in cases of reducing the dimensions.

3x3 is the smallest unit which can be used to compute any kernel size output and seems to be a best fit. If we need 5x5 kernel output, we can convolve with _3x3 twice (3x3 + 3x3 = 18 parameter)_ and if we need 7x7 output, we can convolve using _3x3 thrice (33 + 33 + 3*3 = 27 parameters)_ and so on. And GPUs have accelerated 3x3 operation, so it is much faster to perform the convolution using 3x3 kernel.

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)</br>

### How are kernels initialized?</br>

### What happens during the training of a DNN?</br>
