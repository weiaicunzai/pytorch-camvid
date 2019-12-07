import argparse
import os
import sys

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

import transforms
import utils
from conf import settings
from dataset.camvid import CamVid
from lr_scheduler import WarmUpLR
from metrics import Metrics
from model import UNet




    root_path = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(
        root_path, settings.CHECKPOINT_FOLDER, settings.TIME_NOW)
    log_dir = os.path.join(root_path, settings.LOG_FOLDER, settings.TIME_NOW)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

def get_train_dataloader(data_path, image_size, batch_size, mean, std)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
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
        train_dataset, batch_size=args.b, num_workers=4)

    return train_loader

def get_test_dataloader(data_path, image_size, batch_size, mean, std)
    valid_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD),
    ])

    valid_dataset = CamVid(
        settings.DATA_PATH, 
        'val',
        transforms=train_transforms,
    )
    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.b, num_workers=4)

    return validation_loader

    net = UNet(3, settings.CLASS_NUM)
    net = net.cuda()

    tensor = torch.Tensor(1, 3, *[settings.IMAGE_SIZE] * 2)
    utils.visualize_network(writer, net, tensor)
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    iter_per_epoch = len(train_dataset) / args.b
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.MILESTONES)
    loss_fn = nn.CrossEntropyLoss()


    metrics = Metrics(settings.CLASS_NUM, train_dataset.ignore_index)
    best_iou = 0 
    for epoch in range(1, args.e + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        net.train()

        ious = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            if epoch <= args.warm:
                warmup_scheduler.step()

            optimizer.zero_grad()

            images = images.cuda()
            masks = masks.cuda()
            preds = net(images)

            loss = loss_fn(preds, masks)
            loss.backward()

            optimizer.step()

            preds = preds.argmax(dim=1)
            preds = preds.view(-1).cpu().data.numpy()
            masks = masks.view(-1).cpu().data.numpy()

            metrics.add(preds, masks)
            recall = metrics.recall()
            precision = metrics.precision()
            miou = metrics.iou()

            n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1
            print(('Training Epoch:{epoch} [{trained_samples}/{total_samples}] '
                    'Lr:{lr:0.6f} Loss:{loss:0.4f} mIOU{miou:0.4f} '
                    'Recall:{recall:0.4f} Precision:{precision:0.4f}').format(
                loss=loss.item(),
                epoch=epoch,
                trained_samples=batch_idx * args.b + len(images),
                total_samples=len(train_dataset),
                miou=miou,
                recall=recall,
                precision=precision,
                lr=optimizer.param_groups[0]['lr'],
            ))

            metrics.clear()
            utils.visualize_scalar(
                writer,
                'Train/Loss',
                loss.item(),
                n_iter,
            )
            utils.visulaize_lastlayer(
                writer,
                net,
                n_iter,
            )
            utils.visualize_scalar(
                writer,
                'Train/mIOU',
                miou,
                n_iter,
            )

        utils.visualize_scalar(
            writer, 
            'Train/LearningRate', 
            optimizer.param_groups[0]['lr'], 
            epoch,
        )

        utils.visualize_param_hist(writer, net, epoch)
        net.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(validation_loader):

                images = images.cuda()
                masks = masks.cuda()

                preds = net(images)

                loss = loss_fn(preds, masks)
                test_loss += loss.item()

                preds = preds.argmax(dim=1)
                preds = preds.view(-1).cpu().data.numpy()
                masks = masks.view(-1).cpu().data.numpy()
                metrics.add(preds, masks)
                n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1

        miou = metrics.iou()
        precision = metrics.precision()
        recall = metrics.recall()
        metrics.clear()

        utils.visualize_scalar(
            writer,
            'Test/mIOU',
            miou,
            epoch,
        )

        utils.visualize_scalar(
            writer,
            'Test/Loss',
            test_loss / len(valid_dataset),
            epoch,
        )

        eval_msg = (
            'Test set Average loss: {loss:.4f}, '
            'mIOU: {miou:.4f}, '
            'recall: {recall:.4f}, '
            'precision: {precision:.4f}'
        )

        print(eval_msg.format(
            loss=test_loss / len(valid_dataset),
            miou=miou,
            recall=recall,
            precision=precision
        ))

        if best_iou < miou and epoch > settings.MILESTONES[-1]:
            best_iou = miou
            torch.save(net.state_dict(),
                            checkpoint_path.format(epoch=epoch, type='best'))
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(),
                            checkpoint_path.format(epoch=epoch, type='regular'))

def train_loop_fn(net, train_loader, device, context):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=Flags['lr'],
                        momentum=0.9, weight_decay=1e-4, nesterov=True)

    optimizer = context.getattr_or(
        'optimizer',
        lambda: optim.SGD(net.parameters(), lr=Flags['lr'],
                        momentum=0.9, weight_decay=1e-4, nesterov=True)
    )

    iter_per_epoch = len(train_loader)
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


    metrics = Metrics(settings.CLASS_NUM, train_loader.train_dataset.ignore_index)

    best_iou = 0

    #for epoch in range(1, args.e + 1):
    #    if epoch 
    #    if epoch > args.warm:
    #        train_scheduler.step(epoch)

    net.train()

    ious = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        if warm_scheduler.last_epoch >= iter_per_epoch * args.warm:
            warmup_scheduler.step()

        optimizer.zero_grad()

        images = images.cuda()
        masks = masks.cuda()
        preds = net(images)

        loss = loss_fn(preds, masks)
        loss.backward()

        optimizer.step()

        preds = preds.argmax(dim=1)
        preds = preds.view(-1)
        masks = masks.view(-1)

        metrics.add(preds.cup().data.numpy(), masks.cup().data.numpy())
        miou = metrics.iou()
        metrics.clear()

        print(('Device: {device} [{trained_samples}/{total_samples}]',
               'Lr: {lr:0.5f} Loss{loss:0.4f} mIou{miou:0.4f}').format(
                   device=device,
                   trained_samples=args.b * batch_idx,
                   total_samples=len(train_loader) * args.b,
                   loss=loss.item(),
                   miou=miou,
               )
        ))




#def _mp_fn(rank):
#    device = xm.xla_device()
#    torch.set_default_tensor_type('torch.FloatTensor')
#    train_loader = get_train_dataloader()
#    para_loader = pl.ParallelLoader(train_loader, [device])
#    lr = args.lr * xm.xrt_world_size()


  #global FLAGS
  #FLAGS = flags
  #torch.set_default_tensor_type('torch.FloatTensor')
  #accuracy, data, pred, target = train_resnet18()
  #if rank == 0:
    # Retrieve tensors that are on TPU core 0 and plot.
    #plot_results(data.cpu(), pred.cpu(), target.cpu())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=5,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1,
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

    Flag = {}
    Flag['lr'] = args.lr
    Flag['epoch'] = args.e
    Flag['warm'] = args.warm
    Flag['batch_size'] = args.b
    Flag['milestones'] = settings.MILESTONES

    data_path, image_size, batch_size, mean, std

