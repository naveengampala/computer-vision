<img src="scenarios/media/saccaldic_motion.jpg" align="right" alt="" width="225"/>

# Advacnced Concepts 
- **Advance Convolutions**
- **Attention and Image Augmentation: Depthwise**
- **Pixel Shuffle** 
- **Dilated, Transpose**
- **Channel Attention**


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
    
##  Train and Test Logs

	  0%|          | 0/391 [00:00<?, ?it/s]

Epoch 0

loss=1.3029835224151611 batch_id=390: 100%|██████████| 391/391 [00:15<00:00, 24.81it/s]


Train set: Average loss: 0.0123, Accuracy: 21173/50000 (42.35%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -2.1886, Accuracy: 5241/10000 (52.41%)

Epoch 1

loss=1.2386891841888428 batch_id=390: 100%|██████████| 391/391 [00:15<00:00, 24.47it/s]


Train set: Average loss: 0.0095, Accuracy: 28295/50000 (56.59%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -3.3724, Accuracy: 6276/10000 (62.76%)

Epoch 2

loss=1.1084434986114502 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 24.35it/s]


Train set: Average loss: 0.0084, Accuracy: 31022/50000 (62.04%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -3.9852, Accuracy: 6662/10000 (66.62%)

Epoch 3

loss=0.9563189744949341 batch_id=390: 100%|██████████| 391/391 [00:15<00:00, 24.46it/s]


Train set: Average loss: 0.0077, Accuracy: 32832/50000 (65.66%)

  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.1222, Accuracy: 6668/10000 (66.68%)

Epoch 4

loss=0.9270075559616089 batch_id=390: 100%|██████████| 391/391 [00:15<00:00, 24.68it/s]


Train set: Average loss: 0.0072, Accuracy: 33772/50000 (67.54%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.6931, Accuracy: 7142/10000 (71.42%)

Epoch 5

loss=0.9512995481491089 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 24.23it/s]


Train set: Average loss: 0.0069, Accuracy: 34614/50000 (69.23%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.5787, Accuracy: 7200/10000 (72.00%)

Epoch 6

loss=0.9948743581771851 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.86it/s]


Train set: Average loss: 0.0066, Accuracy: 35137/50000 (70.27%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.8217, Accuracy: 7318/10000 (73.18%)

Epoch 7

loss=0.8459811210632324 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.53it/s]


Train set: Average loss: 0.0063, Accuracy: 35868/50000 (71.74%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.8625, Accuracy: 7480/10000 (74.80%)

Epoch 8

loss=0.9146134257316589 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.07it/s]


Train set: Average loss: 0.0061, Accuracy: 36268/50000 (72.54%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.0781, Accuracy: 7596/10000 (75.96%)

Epoch 9

loss=0.7606831789016724 batch_id=390: 100%|██████████| 391/391 [00:18<00:00, 21.14it/s]


Train set: Average loss: 0.0060, Accuracy: 36550/50000 (73.10%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.1368, Accuracy: 7533/10000 (75.33%)

Epoch 10

loss=0.9202811121940613 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.09it/s]


Train set: Average loss: 0.0059, Accuracy: 36907/50000 (73.81%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.1783, Accuracy: 7659/10000 (76.59%)

Epoch 11

loss=0.6317022442817688 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.03it/s]


Train set: Average loss: 0.0057, Accuracy: 37027/50000 (74.05%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.3921, Accuracy: 7725/10000 (77.25%)

Epoch 12

loss=0.5649533271789551 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.02it/s]


Train set: Average loss: 0.0056, Accuracy: 37448/50000 (74.90%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.6482, Accuracy: 7662/10000 (76.62%)

Epoch 13

loss=0.5654287934303284 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.07it/s]


Train set: Average loss: 0.0055, Accuracy: 37665/50000 (75.33%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.5438, Accuracy: 7735/10000 (77.35%)

Epoch 14

loss=0.6845245361328125 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.40it/s]


Train set: Average loss: 0.0054, Accuracy: 37774/50000 (75.55%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.6981, Accuracy: 7761/10000 (77.61%)

Epoch 15

loss=0.7135276198387146 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.25it/s]


Train set: Average loss: 0.0054, Accuracy: 38040/50000 (76.08%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.6776, Accuracy: 7818/10000 (78.18%)

Epoch 16

loss=0.8259657621383667 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.33it/s]


Train set: Average loss: 0.0052, Accuracy: 38291/50000 (76.58%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.8414, Accuracy: 7821/10000 (78.21%)

Epoch 17

loss=0.7589135766029358 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.16it/s]


Train set: Average loss: 0.0052, Accuracy: 38302/50000 (76.60%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.6223, Accuracy: 7785/10000 (77.85%)

Epoch 18

loss=0.5252959728240967 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.59it/s]


Train set: Average loss: 0.0052, Accuracy: 38473/50000 (76.95%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.8578, Accuracy: 7788/10000 (77.88%)

Epoch 19

loss=0.7093946933746338 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.80it/s]


Train set: Average loss: 0.0051, Accuracy: 38617/50000 (77.23%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.7626, Accuracy: 7811/10000 (78.11%)

Epoch 20

loss=0.4621926248073578 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.83it/s]


Train set: Average loss: 0.0050, Accuracy: 38752/50000 (77.50%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.7677, Accuracy: 7948/10000 (79.48%)

Epoch 21

loss=0.7695800065994263 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.68it/s]


Train set: Average loss: 0.0050, Accuracy: 38826/50000 (77.65%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.0756, Accuracy: 8056/10000 (80.56%)

Epoch 22

loss=0.7181563973426819 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.72it/s]


Train set: Average loss: 0.0050, Accuracy: 38966/50000 (77.93%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.1156, Accuracy: 7935/10000 (79.35%)

Epoch 23

loss=0.6651536226272583 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.75it/s]


Train set: Average loss: 0.0049, Accuracy: 39051/50000 (78.10%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4036, Accuracy: 7971/10000 (79.71%)

Epoch 24

loss=0.4773537516593933 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.76it/s]


Train set: Average loss: 0.0049, Accuracy: 39093/50000 (78.19%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.0348, Accuracy: 8056/10000 (80.56%)

Epoch 25

loss=0.6841117143630981 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.50it/s]


Train set: Average loss: 0.0046, Accuracy: 39641/50000 (79.28%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3500, Accuracy: 8152/10000 (81.52%)

Epoch 26

loss=0.5019469261169434 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.19it/s]


Train set: Average loss: 0.0045, Accuracy: 39881/50000 (79.76%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3410, Accuracy: 8122/10000 (81.22%)

Epoch 27

loss=0.7363773584365845 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.89it/s]


Train set: Average loss: 0.0045, Accuracy: 39957/50000 (79.91%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3463, Accuracy: 8158/10000 (81.58%)

Epoch 28

loss=0.32082101702690125 batch_id=390: 100%|██████████| 391/391 [00:18<00:00, 21.65it/s]


Train set: Average loss: 0.0045, Accuracy: 39966/50000 (79.93%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3897, Accuracy: 8143/10000 (81.43%)

Epoch 29

loss=0.6500608921051025 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 21.76it/s]


Train set: Average loss: 0.0045, Accuracy: 39996/50000 (79.99%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3663, Accuracy: 8164/10000 (81.64%)

Epoch 30

loss=0.6943362355232239 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.59it/s]


Train set: Average loss: 0.0044, Accuracy: 40202/50000 (80.40%)

  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3934, Accuracy: 8149/10000 (81.49%)

Epoch 31

loss=0.49764880537986755 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.75it/s]


Train set: Average loss: 0.0044, Accuracy: 40109/50000 (80.22%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4579, Accuracy: 8166/10000 (81.66%)

Epoch 32

loss=0.6505581140518188 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.55it/s]


Train set: Average loss: 0.0044, Accuracy: 40177/50000 (80.35%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4604, Accuracy: 8158/10000 (81.58%)

Epoch 33

loss=0.5677257180213928 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.68it/s]


Train set: Average loss: 0.0044, Accuracy: 40210/50000 (80.42%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4243, Accuracy: 8168/10000 (81.68%)

Epoch 34

loss=0.6512405276298523 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.84it/s]


Train set: Average loss: 0.0044, Accuracy: 40225/50000 (80.45%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3939, Accuracy: 8175/10000 (81.75%)

Epoch 35

loss=0.5627875924110413 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.24it/s]


Train set: Average loss: 0.0044, Accuracy: 40077/50000 (80.15%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4499, Accuracy: 8175/10000 (81.75%)

Epoch 36

loss=0.44181641936302185 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.56it/s]


Train set: Average loss: 0.0044, Accuracy: 40303/50000 (80.61%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4724, Accuracy: 8185/10000 (81.85%)

Epoch 37

loss=0.576661229133606 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.39it/s]


Train set: Average loss: 0.0044, Accuracy: 40268/50000 (80.54%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4448, Accuracy: 8182/10000 (81.82%)

Epoch 38

loss=0.4060697555541992 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.11it/s]


Train set: Average loss: 0.0044, Accuracy: 40202/50000 (80.40%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4085, Accuracy: 8197/10000 (81.97%)

Epoch 39

loss=0.5688197016716003 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.22it/s]


Train set: Average loss: 0.0043, Accuracy: 40290/50000 (80.58%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.5254, Accuracy: 8187/10000 (81.87%)

Epoch 40

loss=0.5986300706863403 batch_id=390: 100%|██████████| 391/391 [00:17<00:00, 22.92it/s]


Train set: Average loss: 0.0043, Accuracy: 40319/50000 (80.64%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4623, Accuracy: 8197/10000 (81.97%)

Epoch 41

loss=0.6285001039505005 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.28it/s]


Train set: Average loss: 0.0044, Accuracy: 40342/50000 (80.68%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4937, Accuracy: 8218/10000 (82.18%)

Epoch 42

loss=0.6367213726043701 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.28it/s]


Train set: Average loss: 0.0043, Accuracy: 40423/50000 (80.85%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4506, Accuracy: 8190/10000 (81.90%)

Epoch 43

loss=0.521529495716095 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.44it/s]


Train set: Average loss: 0.0043, Accuracy: 40314/50000 (80.63%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4451, Accuracy: 8185/10000 (81.85%)

Epoch 44

loss=0.5996056795120239 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.33it/s]


Train set: Average loss: 0.0043, Accuracy: 40354/50000 (80.71%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4949, Accuracy: 8198/10000 (81.98%)

Epoch 45

loss=0.7322646379470825 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.44it/s]


Train set: Average loss: 0.0043, Accuracy: 40311/50000 (80.62%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4887, Accuracy: 8202/10000 (82.02%)

Epoch 46

loss=0.4716852307319641 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.44it/s]


Train set: Average loss: 0.0043, Accuracy: 40275/50000 (80.55%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.5118, Accuracy: 8196/10000 (81.96%)

Epoch 47

loss=0.5714178085327148 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.22it/s]


Train set: Average loss: 0.0043, Accuracy: 40370/50000 (80.74%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.4216, Accuracy: 8189/10000 (81.89%)

Epoch 48

loss=0.8497797250747681 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.40it/s]


Train set: Average loss: 0.0043, Accuracy: 40392/50000 (80.78%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.5624, Accuracy: 8198/10000 (81.98%)

Epoch 49

loss=0.5186148881912231 batch_id=390: 100%|██████████| 391/391 [00:16<00:00, 23.19it/s]


Train set: Average loss: 0.0043, Accuracy: 40325/50000 (80.65%)



Test set: Average loss: -6.4636, Accuracy: 8219/10000 (82.19%)

## Accuracy by class
	Accuracy of plane : 93 %
	Accuracy of   car : 89 %
	Accuracy of  bird : 75 %
	Accuracy of   cat : 52 %
    Accuracy of  deer : 92 %
    Accuracy of   dog : 69 %
    Accuracy of  frog : 91 %
    Accuracy of horse : 84 %
    Accuracy of  ship : 93 %
    Accuracy of truck : 87 %
    
## Vaidation Plot
![image](https://github.com/naveengampala/computer-vision/blob/main/Season%2007%20-%20Advanced%20Concepts/plots/validation.png)

	

