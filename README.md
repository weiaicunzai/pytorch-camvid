# Pytorch Segmentation

This is a Pytorch implementation for Sementation Networks
All experices run on CamVid dataset


# Camvid dataset
The [original Camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) have 367 training, 233 test images and 101 validation images, total 32 classes. In this project we will use the original training and test images as training set(600 images in total), and the original validation images(101 images) as validation set.We use Camvid dataset provived by [fast.ai](https://course.fast.ai/datasets#image-localization), they already splited the training and validation dataset for us.


Most image segmentation papers group the similar classes in the original Camvid dataset into one class,resulting total 12 classes:
```
'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void'
```
We will do the same.

# Train the Model

## GPU
Model was trained on a P100 GPU, you could just simply run
```
python train.py
```

## TPU
Training on TPU using Pytorch is supported by the Pytorch/xla library, but there is no currently stable release available,
the nightly version is highly unstable, so I'll update my TPU training code after the stable release.

# Supported Model

```
unet
segnet
```
# Result

|Dataset|Network|Parameters|mIOU|
|:-----:|:-----:|:--------:|:--:|
|Camvid|UNet|34.5M|0.6296|
|Camvid|SegNet|29.4M|0.5913|
