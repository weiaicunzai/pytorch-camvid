import argparse

import torch
import torch.nn as nn

import transforms
from conf import settings
import utils
from metric import eval_metrics

#from dataset.camvid import CamVid
#from metrics import Metrics
#from model import UNet



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-weight', type=str, required=True,
                        help='weight file path')
    parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-download', action='store_true', default=False)
    parser.add_argument('-b', type=int, default=1,
                        help='batch size for dataloader')
    args = parser.parse_args()

    test_dataloader = utils.data_loader(args, 'test')
    test_dataset = test_dataloader.dataset
    net = utils.get_model(args.net, 3, test_dataset.class_num)
    net.load_state_dict(torch.load(args.weight))

    net = net.cuda()
    net.eval()

    with torch.no_grad():
        utils.test(
            net,
            test_dataloader,
            settings.IMAGE_SIZE,
            settings.SCALES,
            settings.BASE_SIZE,
            test_dataset.class_num,
            settings.MEAN,
            settings.STD
        )

        #utils.test(
        #    net,
        #    test_dataloader,
        #    settings.IMAGE_SIZE,
        #    [1],
        #    settings.BASE_SIZE,
        #    test_dataset.class_num,
        #    settings.MEAN,
        #    settings.STD
        #)

#    import random
#    random.seed(42)
#    val_dataloader = utils.data_loader(args, 'val')
#    val_dataset = val_dataloader.dataset
#    cls_names = val_dataset.class_names
#    ig_idx = val_dataset.ignore_index
#    iou = 0
#    all_acc = 0
#    acc = 0
#
#    ioud = 0
#    all_accd = 0
#    accd = 0
#    with torch.no_grad():
#        for img, label in val_dataloader:
#            img = img.cuda()
#            b = img.shape[0]
#            label = label.cuda()
#            pred = net(img)
#            pred = pred.argmax(dim=1)
#
#            tmp_all_acc, tmp_acc, tmp_iou = eval_metrics(
#                    pred.detach().cpu().numpy(),
#                    label.detach().cpu().numpy(),
#                    len(cls_names),
#                    ignore_index=ig_idx,
#                    metrics='mIoU',
#                    nan_to_num=-1
#            )
#            tmp_all_accd, tmp_accd, tmp_ioud = dd.eval_metrics(
#                label.detach(),
#                pred.detach(),
#                len(cls_names)
#            )
#
#
#            all_acc += tmp_all_acc * b
#            acc += tmp_acc * b
#            iou += tmp_iou * b
#
#            all_accd += tmp_all_accd * b
#            accd += tmp_accd * b
#            ioud += tmp_ioud * b
#
#        all_acc /= len(val_dataloader.dataset)
#        acc /= len(val_dataloader.dataset)
#        iou /= len(val_dataloader.dataset)

        #print('Iou for each class:')
        #utils.print_eval(cls_names, iou)
        #print('Acc for each class:')
        #utils.print_eval(cls_names, acc)
        #miou = sum(iou) / len(iou)
        #macc = sum(acc) / len(acc)
        #print('Mean acc {:.4f} Mean iou {:.4f}  All Pixel Acc {:.4f}'.format(macc, miou, all_acc))
    #valid_transforms = transforms.Compose([
    #    transforms.Resize(settings.IMAGE_SIZE),
    #    transforms.ToTensor(),
    #    transforms.Normalize(settings.MEAN, settings.STD)
    #])
        #all_accd /= len(val_dataloader.dataset)
        #accd /= len(val_dataloader.dataset)
        #ioud /= len(val_dataloader.dataset)

        #print(accd)
        #print(ioud)
        #print(all_accd)

    #valid_dataset = CamVid(
    #    settings.DATA_PATH,
    #    'val',
    #    valid_transforms
    #)

    #valid_loader = torch.utils.data.DataLoader(
    #    valid_dataset, batch_size=args.b, num_workers=4)

    #metrics = Metrics(valid_dataset.class_num, valid_dataset.ignore_index)

    #loss_fn = nn.CrossEntropyLoss()

    #net = UNet(3, valid_dataset.class_num)
    #net.load_state_dict(torch.load(args.weight))
    #net = net.cuda()

    #net.eval()
    #test_loss = 0
    #with torch.no_grad():
    #    for batch_idx, (images, masks) in enumerate(valid_loader):

    #        images = images.cuda()
    #        masks = masks.cuda()

    #        preds = net(images)

    #        loss = loss_fn(preds, masks)
    #        test_loss += loss.item()

    #        preds = preds.argmax(dim=1)
    #        preds = preds.view(-1).cpu().data.numpy()
    #        masks = masks.view(-1).cpu().data.numpy()
    #        metrics.add(preds, masks)

    #        print('iteration: {}, loss: {:.4f}'.format(batch_idx, loss))

    #test_loss = test_loss / len(valid_loader)
    #miou = metrics.iou()
    #precision = metrics.precision()
    #recall = metrics.recall()
    #metrics.clear()


    #print(('miou: {miou:.4f}, precision: {precision:.4f}, '
    #       'recall: {recall:.4f}, average loss: {loss:.4f}').format(
    #    miou=miou,
    #    precision=precision,
    #    recall=recall,
    #    loss=test_loss
    #))