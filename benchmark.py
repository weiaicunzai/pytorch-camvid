import os
import time

import torch
import numpy as np

import transforms
from conf import settings
#from dataset.camvid_lmdb import CamVid
from dataset.camvid import CamVid

if __name__ == '__main__':


    train_dataset = CamVid(
        'data',
        image_set='train',
        download=True
    )
    valid_dataset = CamVid(
        'data',
        image_set='val',
        download=True
    )

    train_transforms = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.RandomRotation(15, fill=train_dataset.ignore_index),
            transforms.RandomGaussianBlur(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(settings.MEAN, settings.STD),
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD),
    ])

    train_dataset.transforms = train_transforms
    valid_dataset.transforms = valid_transforms

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, num_workers=4, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, num_workers=4)


    count = 0
    start = time.time()
    for epoch in range(500):

        for image, mask in train_dataset:


            #images = images.cuda()
            #masks = masks.cuda()
            #count += (batch_idx + 1) * args.b
            count += 1

            #if count % (args.b * 300) == 0:
            if count % 1000 == 0:
                finish = time.time()
                #print('total', count, 'time:', round(finish - start, 2), 's','count / second', int(count / (finish - start)))
                total_time = finish - start
                print('total {} samples, total {:.2f}s, average {:.0f} samples/sec'.format( \
                    count, total_time, count / total_time))

    finish = time.time()
    total_time = finish - start
    print('total {} samples, total {:.2f}s, average {:.0f} samples/sec'.format( \
            count, total_time, count / total_time))

