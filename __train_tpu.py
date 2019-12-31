import argparse
import os
import time
import sys
from statistics import mean

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils

import transforms
import utils
from conf import settings
from dataset.camvid import CamVid
from metrics import Metrics
from model import UNet

import torch_xla.debug.metrics as met

print(met.metrics_report())


def get_train_dataloader(data_path, image_size, batch_size, mean, std):
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = CamVid(
        data_path,
        'train',
        transforms=train_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.b, num_workers=4, shuffle=True)

    return train_loader

def get_test_dataloader(data_path, image_size, batch_size, mean, std):
    valid_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD),
    ])

    valid_dataset = CamVid(
        settings.DATA_PATH,
        'val',
        transforms=valid_transforms,
    )

    print(len(valid_dataset))
    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.b, num_workers=4, shuffle=True)

    return validation_loader


def train_loop_fn(net, train_loader, device, context):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=1e-4, nesterov=True)

    optimizer = context.getattr_or(
        'optimizer',
        lambda: optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=1e-4, nesterov=True)
    )

    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    warm_scheduler = context.getattr_or(
        'warm_scheduler',
        lambda: warmup_scheduler,
    )

    train_scheduler = context.getattr_or(
        'train_scheduler',
        lambda: optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=settings.MILESTONES)
    )

    net.train()

    count = 0
    total_loss = 0
    for batch_idx, (images, masks) in enumerate(train_loader):

        count += 1
        if warm_scheduler.last_epoch <= iter_per_epoch * args.warm:
            warmup_scheduler.step()

        optimizer.zero_grad()
        #print(images.shape)
        preds = net(images)
        loss = loss_fn(preds, masks)

        loss.backward()
        xm.optimizer_step(optimizer)

        print('Epoch: {epoch}, device: {device}, loss: {loss:0.4f}, lr: {lr:0.6f}'.format(
            epoch=epoch,
            device=device,
            loss=loss,
            lr=optimizer.param_groups[0]['lr'],
        ))

        total_loss += loss
        #with torch.no_grad():
        #    preds = preds.argmax(dim=1)
        #    preds = preds.view(-1)
        #    masks = masks.view(-1)

        #metrics.add(preds.cpu().data.numpy(), masks.cpu().data.numpy())
        #miou = metrics.iou()
        #metrics.clear()
        #print(('Device: {device} [{trained_samples}]'
        #       'Lr: {lr:0.5f} Loss{loss:0.4f} mIou{miou:0.4f}').format(
        #           device=device,
        #           trained_samples=args.b * batch_idx,
        #           #total_samples=len(train_loader) * args.b,
        #           lr=optimizer.param_groups[0]['lr'],
        #           loss=loss.item(),
        #           miou=miou,
        #       )
        #)

    #if warm_scheduler.last_epoch > iter_per_epoch * args.warm:
    #    if train_scheduler.last_epoch < args.warm:
    #        train_scheduler.step(args.warm)

    #    train_scheduler.step()
    return total_loss / (batch_idx + 1)

def test_loop_fn(net, test_loader, device, context):
    loss_fn = nn.CrossEntropyLoss()


    pred_res = []
    mask_res = []
    total_loss = 0
    count = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):

            preds = net(images)
            loss = loss_fn(preds, masks)

            total_loss += loss
            count += 1

            preds = preds.argmax(dim=1)
            pred_res.append(preds.cpu().data.numpy())
            mask_res.append(masks.cpu().data.numpy())

    return pred_res, mask_res, total_loss.cpu().data.numpy() / count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=5,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('-e', type=int, default=150, help='training epoches')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    args = parser.parse_args()

    train_data_loader = get_train_dataloader(
        settings.DATA_PATH,
        settings.IMAGE_SIZE,
        args.b,
        settings.MEAN,
        settings.STD,
    )

    test_data_loader = get_test_dataloader(
        settings.DATA_PATH,
        settings.IMAGE_SIZE,
        args.b,
        settings.MEAN,
        settings.STD,
    )

    Flag = {}
    Flag['lr'] = args.lr
    Flag['epoch'] = args.e
    Flag['warm'] = args.warm
    Flag['batch_size'] = args.b
    Flag['milestones'] = settings.MILESTONES
    Flag['ignore_idx'] = train_data_loader.dataset.ignore_index

    len(train_data_loader.dataset)
    net = UNet(3, train_data_loader.dataset.class_num)
    devices = (
      xm.get_xla_supported_devices())
    print(devices)
    net = dp.DataParallel(net, device_ids=devices)

    iter_per_epoch = len(train_data_loader) / 8

    best_iou = 0
    for epoch in range(1, args.e + 1):
        print('training epoch {}'.format(epoch))
        t1 = time.time()

        net(train_loop_fn, train_data_loader)

        print(time.time() - t1)

        #result = net(test_loop_fn, test_data_loader)
        #pred_res = np.array([res[0] for res in result])
        #mask_res = np.array([res[1] for res in result])
        #loss = np.array([res[2] for res in result])

        #t1 = time.time()
        #metrics = Metrics(settings.CLASS_NUM, train_data_loader.dataset.ignore_index)
        #pred_res = np.array(pred_res).reshape(-1)
        #mask_res = np.array(mask_res).reshape(-1)
        #loss = np.array(loss).reshape(-1)
        #metrics.add(pred_res, mask_res)
        #miou = metrics.iou()

        #loss = loss.mean()
        #print("Test: miou: {miou:4f} loss: {loss:4f} evaluate time: {time:4f}".format(
        #    miou=miou,
        #    loss=loss,
        #    time=time.time() - t1,
        #))

        #if best_iou < miou and epoch > settings.MILESTONES[-1]:
        #    state_dict = net.models[0].to('cpu').state_dict()
        #    best_iou = miou
        #    torch.save(state_dict,
        #                    checkpoint_path.format(epoch=epoch, type='best'))
        #    continue

        #if not epoch % settings.SAVE_EPOCH:
        #    state_dict = net.models[0].to('cpu').state_dict()
        #    torch.save(state_dict,
        #                    checkpoint_path.format(epoch=epoch, type='regular'))
