# Pytorch Segmentation

This is a Pytorch implementation for Sementation Networks
All experices run on CamVid dataset


# Download Camvid dataset

Camvid dataset:```https://course.fast.ai/datasets```
or simply 
```
wget https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz
```
and change the data path in conf/settings.py

Camvid is a small image segmentation dataset(600 train, 101 val), contains 
32 classes, we will group similar classes into one class, turn 32 classes 
into 12 classes. Small dataset with more classes will strongly influence 
the performance of a network.


# Train the Model

## GPU
Model was trained on a google colab P100

# TPU