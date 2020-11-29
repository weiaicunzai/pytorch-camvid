import os
import time

import torch
import numpy as np

import transforms
from conf import settings
#from dataset.camvid_lmdb import CamVid
#from dataset.camvid import CamVid
from dataset.voc2012 import VOC2012Aug
from torchvision.datasets import SBDataset

if __name__ == '__main__':


    #train_dataset = CamVid(
    #    'data',
    #    image_set='train',
    #    download=True
    #)
    #valid_dataset = CamVid(
    #    'data',
    #    image_set='val',
    #    download=True
    #)
    #train_dataset = SBDataset('/data/by/pytorch-camvid/tmp/benchmark_RELEASE/dataset/', image_set='train')
    #train_dataset = VOC2012Aug('/data/by/datasets/voc2012_aug', image_set='train')
    train_dataset = VOC2012Aug('voc_aug', image_set='train')
    print(len(train_dataset))
    #valid_dataset = VOC2012Aug('/data/by/datasets/voc2012_aug', image_set='val')
    valid_dataset = VOC2012Aug('voc_aug', image_set='val')
    print(len(valid_dataset))
    #train_dataset = SBDataset('tmp', image_set='train_noval')

    train_transforms = transforms.Compose([
            #transforms.Resize(settings.IMAGE_SIZE),
            transforms.RandomCrop(513, pad_if_needed=True),

            #transforms.RandomRotation(15, fill=train_dataset.ignore_index),
            #transforms.RandomRotation(15, fill=0),
            #transforms.RandomGaussianBlur(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(0.4, 0.4),
            #transforms.ToTensor(),
            #transforms.Normalize(settings.MEAN, settings.STD),
    ])

    valid_transforms = transforms.Compose([
        #transforms.Resize(settings.IMAGE_SIZE),

        transforms.RandomCrop(513, pad_if_needed=True),

        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD),
    ])

    train_dataset.transforms = train_transforms
    valid_dataset.transforms = valid_transforms

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, num_workers=4)

    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, num_workers=4)


    count = 0
    start = time.time()
    for epoch in range(500):

        print('here')
        for idx, (image, mask) in enumerate(train_loader):


            #images = images.cuda()
            #masks = masks.cuda()
            #count += (batch_idx + 1) * args.b
            count += 8

            #if count % (args.b * 300) == 0:
            if count % (1000 * 8) == 0:
                finish = time.time()
                #print('total', count, 'time:', round(finish - start, 2), 's','count / second', int(count / (finish - start)))
                total_time = finish - start
                print('total {} samples, total {:.2f}s, average {:.0f} samples/sec'.format( \
                    count, total_time, count / total_time))

    finish = time.time()
    total_time = finish - start
    print('total {} samples, total {:.2f}s, average {:.0f} samples/sec'.format( \
            count, total_time, count / total_time))

