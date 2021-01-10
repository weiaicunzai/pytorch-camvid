import argparse
import os
import time
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import transforms
import utils
from conf import settings
from dataset.camvid import CamVid
from dataset.voc2012 import VOC2012Aug
#from dataset.camvid_lmdb import CamVid
from lr_scheduler import PolyLR
from metric import eval_metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.007,
                        help='initial learning rate')
    parser.add_argument('-e', type=int, default=120, help='training epoches')
    parser.add_argument('-wd', type=float, default=0, help='training epoches')
    parser.add_argument('-resume', type=bool, default=False, help='if resume training')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-download', action='store_true', default=False,
        help='whether to download camvid dataset')
    parser.add_argument('-gpu', action='store_true', default=False, help='whether to use gpu')
    args = parser.parse_args()

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

    #train_dataset = CamVid(
    #    'data',
    #    image_set='train',
    #    download=args.download
    #)
    #valid_dataset = CamVid(
    #    'data',
    #    image_set='val',
    #    download=args.download
    #)
    #train_dataset = VOC2012Aug(
    #    'voc_aug',
    #    image_set='train'
    #)
    #valid_dataset = VOC2012Aug(
    #    'voc_aug',
    #    image_set='val'
    #)
    print()

    #train_transforms = transforms.Compose([
    #        transforms.RandomHorizontalFlip(),
    #        transforms.RandomRotation(15, fill=train_dataset.ignore_index),
    #        transforms.RandomScaleCrop(settings.IMAGE_SIZE),
    #        transforms.RandomGaussianBlur(),
    #        transforms.ColorJitter(0.4, 0.4),
    #        transforms.ToTensor(),
    #        transforms.Normalize(settings.MEAN, settings.STD),
    #])

    #valid_transforms = transforms.Compose([
    #    transforms.RandomScaleCrop(settings.IMAGE_SIZE),
    #    transforms.ToTensor(),
    #    transforms.Normalize(settings.MEAN, settings.STD),
    #])

    #train_dataset.transforms = train_transforms
    #valid_dataset.transforms = valid_transforms

    #train_loader = torch.utils.data.DataLoader(
    #        train_dataset, batch_size=args.b, num_workers=4, shuffle=True, pin_memory=True)

    #validation_loader = torch.utils.data.DataLoader(
    #    valid_dataset, batch_size=args.b, num_workers=4, pin_memory=True)
    train_loader = utils.data_loader(args, 'train')
    train_dataset = train_loader.dataset

    validation_loader = utils.data_loader(args, 'val')
    valid_dataset = validation_loader.dataset

    net = utils.get_model(args.net, 3, train_dataset.class_num)

    if args.resume:
        weight_path = utils.get_weight_path(
            os.path.join(root_path, settings.CHECKPOINT_FOLDER))
        print('Loading weight file: {}...'.format(weight_path))
        net.load_state_dict(torch.load(weight_path))
        print('Done loading!')

    if args.gpu:
        net = net.cuda()

    tensor = torch.Tensor(1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    utils.visualize_network(writer, net, tensor)

    #optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(train_dataset) / args.b

    max_iter = args.e * len(train_loader)
    train_scheduler = PolyLR(optimizer, max_iter=max_iter, power=0.9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.ignore_index)

    best_iou = 0

    trained_epochs = 0

    if args.resume:
        trained_epochs = int(
            re.search('([0-9]+)-(best|regular).pth', weight_path).group(1))
        #train_scheduler.step(trained_epochs * len(train_loader))

    for epoch in range(trained_epochs + 1, args.e + 1):
        start = time.time()

        net.train()

        ious = 0
        batch_start = time.time()
        total_load_time = 0
        for batch_idx, (images, masks) in enumerate(train_loader):

            batch_finish = time.time()

            optimizer.zero_grad()

            if args.gpu:
                images = images.cuda()
                masks = masks.cuda()

            preds = net(images)

            loss = loss_fn(preds, masks)
            loss.backward()

            optimizer.step()

            print(('Training Epoch:{epoch} [{trained_samples}/{total_samples}] '
                    #'Lr:{lr:0.6f} Loss:{loss:0.4f} Beta1:{beta:0.4f} Time:{time:0.2f}s').format(
                    'Lr:{lr:0.8f} Loss:{loss:0.4f} Data loading time:{time:0.4f}s').format(
                loss=loss.item(),
                epoch=epoch,
                trained_samples=batch_idx * args.b + len(images),
                total_samples=len(train_dataset),
                lr=optimizer.param_groups[0]['lr'],
                #beta=optimizer.param_groups[0]['betas'][0],
                time=batch_finish - batch_start
            ))
            train_scheduler.step()

            total_load_time += batch_finish - batch_start

            n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1
            utils.visulaize_lastlayer(
                writer,
                net,
                n_iter,
            )

            batch_start = time.time()

        utils.visualize_scalar(
            writer,
            'Train/LearningRate',
            optimizer.param_groups[0]['lr'],
            epoch,
        )

        #utils.visualize_scalar(
        #    writer,
        #    'Train/Beta1',
        #    optimizer.param_groups[0]['betas'][0],
        #    epoch,
        #)
        #print(total_load_time, total_training)

        utils.visualize_param_hist(writer, net, epoch)

        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')

        finish = time.time()
        total_training = finish - start
        print(('Total time for training epoch {} : {:.2f}s, '
               'total time for loading data: {:.2f}s, '
               '{:.2f}% time used for loading data').format(
            epoch,
            total_training,
            total_load_time,
            total_load_time / total_training * 100
        ))

        net.eval()
        test_loss = 0.0

        test_start = time.time()
        iou = 0
        all_acc = 0
        acc = 0
        best_iou = 0

        cls_names = valid_dataset.class_names
        ig_idx = valid_dataset.ignore_index
        with torch.no_grad():
            for images, masks in validation_loader:

                if args.gpu:
                    images = images.cuda()
                    masks = masks.cuda()

                preds = net(images)

                loss = loss_fn(preds, masks)
                test_loss += loss.item()

                preds = preds.argmax(dim=1)
                #tmp_all_acc, tmp_acc, tmp_mean_iou = eval_metrics(
                #    , masks.detach().cpu().numpy(), len(cls_names), ig_idx
                #)
                tmp_all_acc, tmp_acc, tmp_iou = eval_metrics(
                    preds.detach().cpu().numpy(),
                    masks.detach().cpu().numpy(),
                    len(cls_names),
                    ignore_index=ig_idx,
                    metrics='mIoU',
                    nan_to_num=-1
                )

                all_acc += tmp_all_acc * len(images)
                acc += tmp_acc * len(images)
                iou += tmp_iou * len(images)

        all_acc /= len(validation_loader.dataset)
        acc /= len(validation_loader.dataset)
        iou /= len(validation_loader.dataset)
        test_finish = time.time()
        print('Evaluation time comsumed:{:.2f}s'.format(test_finish - test_start))
        print('Iou for each class:')
        utils.print_eval(cls_names, iou)
        print('Acc for each class:')
        utils.print_eval(cls_names, acc)
        #print('%, '.join([':'.join([str(n), str(round(i, 2))]) for n, i in zip(cls_names, iou)]))
        #iou = iou.tolist()
        #iou = [i for i in iou if iou.index(i) != ig_idx]
        miou = sum(iou) / len(iou)
        print('Epoch {}  Mean iou {:.4f}  All Pixel Acc {:.4f}'.format(epoch, miou, all_acc))
        #print('%, '.join([':'.join([str(n), str(round(a, 2))]) for n, a in zip(cls_names, acc)]))
        #print('All acc {:.2f}%'.format(all_acc))

        utils.visualize_scalar(
            writer,
            'Test/mIOU',
            miou,
            epoch,
        )

        utils.visualize_scalar(
            writer,
            'Test/Acc',
            all_acc,
            epoch,
        )

        utils.visualize_scalar(
            writer,
            'Test/Loss',
            test_loss / len(valid_dataset),
            epoch,
        )

        if best_iou < miou and epoch > args.e // 2:
            best_iou = miou
            torch.save(net.state_dict(),
                            checkpoint_path.format(epoch=epoch, type='best'))
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(),
                            checkpoint_path.format(epoch=epoch, type='regular'))
