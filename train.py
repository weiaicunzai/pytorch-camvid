
import os
import argparse
import torch
import torch.nn as nn

import transforms
from model import UNet
from torch.utils.data.sampler import SubsetRandomSampler
from conf import settings

from lr_scheduler import WarmUpLR
from dataset import TableBorder





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
    parser.add_argument('-e', type=int, default=50, help='training epoches')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    parser.add_argument('-c', type=int, default=2, help='number of class')

    args = parser.parse_args()

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, settings.TIME_NOW)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(settings.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD)
    ])

    valid_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = TableBorder(settings.DATA_PATH, transforms=train_transforms)
    valid_dataset = TableBorder(settings.DATA_PATH, transforms=train_transforms)

    split = int(0.2 * len(train_dataset))

    indices = list(range(len(valid_dataset)))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, 
                                           sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.b,
                                                sampler=valid_sampler)

    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES)
    loss_fn = nn.BCELoss()

    net = UNet(3, args.c)
    for epoch in range(1, args.e + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        net.train()

        for batch_idx, (images, masks) in  enumerate(train_loader):

            images = images.cuda()
            masks = masks.cuda()

            preds = net(images)

            loss = loss_fn(preds, masks)
            loss.backward()

            optimizer.step()

        


        
