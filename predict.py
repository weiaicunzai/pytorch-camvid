import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import sys
import numpy as np

import transforms
from model import UNet
from torch.utils.data.sampler import SubsetRandomSampler
from conf import settings

from lr_scheduler import WarmUpLR
from dataset import TableBorder
from utils import get_iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=3,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('-e', type=int, default=50, help='training epoches')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    parser.add_argument('-c', type=int, default=2, help='number of class')

    args = parser.parse_args()

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, settings.TIME_NOW)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    valid_transforms = transforms.Compose([
        transforms.Resize(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD)
    ])


    valid_dataset = TableBorder(
        settings.DATA_PATH, transforms=valid_transforms)





    net = UNet(3, args.c)
    #loss_fn = nn.BCELoss()
    loss_fn = nn.MSELoss()
    net.load_state_dict(torch.load("/home/xuanhua/baiyu/UNet-Pytorch/checkpoints/2019-08-30T13:22:21.683626/40-best.pth"))
    net = net.cuda()

    net.eval()

    min_iou = 1


            #img = images[0, :, :, :].numpy()
            #m = masks[0, :, :, :].numpy()

            #img = images[0, :, :, :].transpose(0, 2).numpy()
            #m = masks[0, :, :, :].transpose(0, 2).numpy()
            
            
            #print(img.shape)

            #print(m.shape)

            #cv2.imwrite("img.jpeg", img)
            #cv2.imwrite('test1.jpeg', m[:, :, 0] * 255)
            #cv2.imwrite('test2.jpeg', m[:, :, 1] * 255)
            #sys.exit(0)
    image , mask = valid_dataset[140]
    #print(image.size())
    #print(mask.size())


    image = image.unsqueeze(0)
    print(image.size())

    image = image.cuda()
    mask = mask.cuda()
    preds = net(image)



    print(loss_fn(preds, mask))
    print(preds.size())
    #print(preds.squeeze().size())
    preds = preds.squeeze()
    preds = preds.transpose(0, 2)
    preds = preds.transpose(0, 1)
    preds = preds.cpu().detach().numpy()
    print(preds.shape)

    mask = mask.squeeze().transpose(0, 2)
    mask = mask.squeeze().transpose(0, 1)
    mask = mask.cpu().detach().numpy()
    cv2.imwrite("mask1.jpg", mask[:, :, 0] * 255)
    cv2.imwrite("mask2.jpg", mask[:, :, 1] * 255)

    image = image.squeeze().transpose(0, 2)
    image = image.squeeze().transpose(0, 1)
    image = image.cpu().detach().numpy()

    image *= settings.STD
    image += settings.MEAN
    
    preds1 = preds[:, :, 0]
    preds2 = preds[:, :, 1]
    preds1 = preds1 > 0.5
    preds2 = preds2 > 0.5

    cv2.imwrite("preds1.jpg", preds1 * 255)
    cv2.imwrite("preds2.jpg", preds2 * 255)
    cv2.imwrite("save.jpg", image * 255)