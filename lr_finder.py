import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import transforms
from conf import settings
from dataset.camvid import CamVid
from lr_scheduler import ExponentialLR
import utils


def lr_finder(train_loader: DataLoader,
              net: nn.Module,
              start_lr: float = 1e-7,
              end_lr: float = 10,
              num_it: int = 100,
              stop_div: bool = True,
              smooth_f: float = 0.05,
              weight_decay: float = 0
              ):
    """Performs the learning rate range test.
    Arguments:
        train_loader (torch.utils.data.DataLoader): the training set data laoder.
        start_lr: the minimum learning rate to start: Default: 1e-7
        end_lr (float, optional): the maximum learning rate to test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        stop_div (bool, optional): the test is stopped when the loss diverges.
        smooth_f (float, optional): the loss smoothing factor within interval
            [0, 1]. Disabled if set to 0, otherwise the loss is smoothed using
            exponential smoothing. Details: 0.05.
        weight_decay: (float, optional): weight_decay factor

    Returns:
        loss (numpy.array): loss for each iteration
        lr (numpy.array): learning rate for each iteration
    """

    optimizer = optim.AdamW(net.parameters(), lr=start_lr,
                            weight_decay=weight_decay)
    exponetial_scheduler = ExponentialLR(optimizer, args.end_lr, args.num_it)
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.ignore_index)

    losses = []
    lrs = []
    stop = False
    count = 0

    for _ in range(1, args.num_it):

        if stop:
            break

        net.train()
        for images, masks in train_loader:

            count += 1
            if count > args.num_it:
                stop = True
                break

            optimizer.zero_grad()

            if args.gpu:
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

            if count != 1:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            losses.append(loss)
            lrs.append(optimizer.param_groups[0]['lr'])

    # plot the result
    loss = np.array(losses[args.skip_start: -args.skip_end])
    lr = np.array(lrs[args.skip_start: -args.skip_end])

    return loss, lr


def plot(loss, lr, skip_start=10, skip_end=5, image_name='lr_finder.jpg'):
    """Draw learning range test result

    Args:
        loss (numpy.array): loss data for each iteration
        lr (numpy.array): learning rate data for each iteration
        skip_start: number of iterations to trim from start
        skip_end: number of iterations to trim from end
        image_name: image path
    """

    plt.plot(lr, loss)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig('lr_finder.jpeg')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=5,
                        help='batch size for dataloader')
    parser.add_argument('-start_lr', type=float, default=1e-7,
                        help='initial learning rate')
    parser.add_argument('-end_lr', type=float, default=10,
                        help='initial learning rate')
    parser.add_argument('-stop_div', type=bool, default=True,
                        help='stops when loss diverges')
    parser.add_argument('-num_it', type=int, default=100,
                        help='number of iterations')
    parser.add_argument('-skip_start', type=int, default=10,
                        help='number of batches to trim from the start')
    parser.add_argument('-skip_end', type=int, default=5,
                        help='number of batches to trim from the end')
    parser.add_argument('-weight_decay', type=float,
                        default=0, help='weight decay factor')
    parser.add_argument('-net', type=str, required=True, help='network name')
    parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-download', action='store_true', default=False,
        help='whether to download camvid dataset')
    parser.add_argument('-gpu', action='store_true', default=False, help='whether to use gpu')


    args = parser.parse_args()

    #train_dataset = CamVid(
    #    settings.DATA_PATH,
    #    'train'
    #)

    #train_transforms = transforms.Compose([
    #    transforms.RandomRotation(value=train_dataset.ignore_index),
    #    transforms.RandomScale(value=train_dataset.ignore_index),
    #    transforms.RandomGaussianBlur(),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ColorJitter(),
    #    transforms.Resize(settings.IMAGE_SIZE),
    #    transforms.ToTensor(),
    #    transforms.Normalize(settings.MEAN, settings.STD),
    #])

    #train_dataset.transforms = train_transforms

    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=args.b, num_workers=4)
    train_loader = utils.data_loader(args, 'train')
    train_dataset = train_loader.dataset

    net = utils.get_model(args.net, 3, train_dataset.class_num)
    if args.gpu:
        net = net.cuda()

    loss, lr = lr_finder(train_loader, net, start_lr=args.start_lr,
                         end_lr=args.end_lr, stop_div=args.stop_div,
                         weight_decay=args.weight_decay)
    plot(loss, lr, skip_start=args.skip_start, skip_end=args.skip_end)
