# Pytorch Segmentation

This is a Pytorch implementation for Sementation Networks
All experices run on CamVid dataset


# Prepare Camvid dataset
   The [original Camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) have
   367 training, 233 testing images and 101 validation images, total 32 classes. We will use the
   training and validation images as training set(468 images), and test images(233 images) as test set.
   Most image segmentation papers group the similar classes in the original Camvid dataset into one class,
   resulting total 12 classes:
   ```
   'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void'
   ```
   We will do the same. The group stage has already been taken care of by the code I wrote. You do not need
   to do anything about this. Here is what you need to do:

1. Download Camvid dataset

   Download Camvid dataset form [fast ai website](https://course.fast.ai/datasets):
   or simply:
   ```
   wget https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz
   ```
   extract it, then the file structure inside Camvid folder should look like this:
   ```
   ├── codes.txt
   ├── images
   ├── labels
   └── valid.txt
   ```
2. Replace valid.txt

   Replace the valid.txt file in Camvid dataset you just downloaded from [fast ai website](https://course.fast.ai/datasets)
   with [the one in my github repo](dataset/valid.txt). We will use the original test set(233 images) as our test set.

3. Change the Camvid folder path

   Remeber to change the folder path in conf/settings.py

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

u-net