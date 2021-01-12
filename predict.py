import os
import argparse
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from model import UNet
from conf import settings
from PIL import Image



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.add_argument('-img', type=str, required=True,
                        help='image path to predict')

    args = parser.add_argument('-weight', type=str, required=True,
                        help='weight file path')

    args = parser.add_argument('-c', type=int, default=32,
                        help='class number')

    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD)
    ])


    src = cv2.imread(args.img)
    image = Image.fromarray(src)
    image = data_transforms(image)
    image = image.unsqueeze(0)
    image = image.cuda()


    net = UNet(3, args.c)
    net.load_state_dict(torch.load(args.weight))
    net = net.cuda()

    net.eval()

    with torch.no_grad():
        preds = net(image)
        preds = torch.argmax(preds, dim=1)
        preds = preds.cpu().data.numpy()
        preds = preds.squeeze(0)
        preds = preds.argmax(dim=1)



    preds = cv2.resize(preds, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('src.jpg', src)
    cv2.imwrite('predict.jpg', preds)