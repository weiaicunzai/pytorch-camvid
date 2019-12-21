import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import transforms
from conf import settings
from dataset.camvid import CamVid
from lr_scheduler import ExponentialLR
from model import UNet

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    parser.add_argument('-start_lr', type=float, default=1e-7,
                        help='initial learning rate')
    parser.add_argument('-end_lr', type=float, default=10,
                        help='initial learning rate')
    parser.add_argument('-stop_div', type=bool, default=True,
                        help='stops when loss diverges')
    parser.add_argument('-num_it', type=int, default=100, help='number of iterations')
    parser.add_argument('-skip_start', type=int, default=10, help='number of batches to trim from the start')
    parser.add_argument('-skip_end', type=int, default=5, help='number of batches to trim from the end')
    args = parser.parse_args()


    train_dataset = CamVid(
        settings.DATA_PATH, 
        'train'
    )

    train_transforms = transforms.Compose([
        transforms.RandomRotation(value=train_dataset.ignore_index),
        transforms.RandomScale(value=train_dataset.ignore_index),
        transforms.RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.Resize(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD),
    ])

    train_dataset.transforms = train_transforms

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.b, num_workers=4)

    net = UNet(3, train_dataset.class_num)
    net = net.cuda()

    
    optimizer = optim.SGD(net.parameters(), lr=args.start_lr,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    exponetial_scheduler = ExponentialLR(optimizer, args.end_lr, args.num_it)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    lrs = []
    stop = False
    count = 0

    for epoch in range(1, args.num_it):

        if stop:
            break

        net.train()
        for batch_idx, (images, masks) in enumerate(train_loader):

            count += 1
            if count > args.num_it:
                stop = True
                break

            optimizer.zero_grad()

            images = images.cuda()
            masks = masks.cuda()
            preds = net(images)

            loss = loss_fn(preds, masks)
            loss.backward()

            if torch.isnan(loss):
                stop = True
                break

            optimizer.step()
            exponetial_scheduler.step()

            print('iteration: {}, lr: {:08f}, loss: {:04f}'.format(
                count, optimizer.param_groups[0]['lr'], loss))

            losses.append(loss)
            lrs.append(optimizer.param_groups[0]['lr'])

    # plot the result
    loss = np.numpy(losses[args.skip_start: -args.skip_end])
    lr = np.numpy(lrs[args.skip_start: -args.skip_start])

    plt.plot(lr, loss)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.imsave('lr_finder.jpeg')
