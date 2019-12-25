import argparse

import torch
import torch.nn as nn

import transforms
from conf import settings
from dataset.camvid import CamVid
from metrics import Metrics
from model import UNet



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    args = parser.add_argument('-weight', type=str, required=True,
                        help='weight file path')
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')

    args = parser.parse_args()

    valid_transforms = transforms.Compose([
        transforms.Resize(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD)
    ])

    valid_dataset = CamVid(
        settings.DATA_PATH,
        'val',
        valid_transforms
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.b, num_workers=4)

    metrics = Metrics(valid_dataset.class_num, valid_dataset.ignore_index)

    loss_fn = nn.CrossEntropyLoss()

    net = UNet(3, valid_dataset.class_num)
    net.load_state_dict(torch.load(args.weight))
    net = net.cuda()

    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(valid_loader):

            images = images.cuda()
            masks = masks.cuda()

            preds = net(images)

            loss = loss_fn(preds, masks)
            test_loss += loss.item()

            preds = preds.argmax(dim=1)
            preds = preds.view(-1).cpu().data.numpy()
            masks = masks.view(-1).cpu().data.numpy()
            metrics.add(preds, masks)

            print('iteration: {}, loss: {:.4f}'.format(batch_idx, loss))

    test_loss = test_loss / len(valid_loader)
    miou = metrics.iou()
    precision = metrics.precision()
    recall = metrics.recall()
    metrics.clear()


    print(('miou: {miou:.4f}, precision: {precision:.4f}, '
           'recall: {recall:.4f}, average loss: {loss:.4f}').format(
        miou=miou,
        precision=precision,
        recall=recall,
        loss=test_loss
    ))