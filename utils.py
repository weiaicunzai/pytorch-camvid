
import os
import glob
import re
import random

import numpy as np
import torch
import cv2

import transforms
from dataset import CamVid, VOC2012Aug
from conf import settings


@torch.no_grad()
def visualize_network(writer, net, tensor):
    tensor = tensor.to(next(net.parameters()).device)
    writer.add_graph(net, tensor)

def _get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur

    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            last_layer_weights = para
        if 'bias' in name:
            last_layer_bias = para

    return last_layer_weights, last_layer_bias

def visulaize_lastlayer(writer, net, n_iter):
    weights, bias = _get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights', weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias', bias.grad.norm(), n_iter)

def visualize_scalar(writer, name, scalar, n_iter):
    """visualize scalar"""
    writer.add_scalar(name, scalar, n_iter)

def visualize_param_hist(writer, net, n_iter):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        #writer.add_histogram("{}/{}".format(layer, attr), param.detach().cpu().numpy(), n_iter)
        writer.add_histogram("{}/{}".format(layer, attr), param, n_iter)

def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset

    Returns:
        return: mean and std of this dataset
    """

    mean_r = 0
    mean_g = 0
    mean_b = 0

    #opencv BGR channel
    for img, _ in dataset:
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std

def get_weight_path(checkpoint_path):
    """return absolute path of the best performance
    weight file path according to file modification
    time.

    Args:
        checkpoint_path: checkpoint folder

    Returns:
        absolute absolute path for weight file
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    weight_files = glob.glob(os.path.join(
        checkpoint_path, '*', '*.pth'), recursive=True)

    best_weights = []
    for weight_file in weight_files:
        m = re.search('.*[0-9]+-best.pth', weight_file)
        if m:
            best_weights.append(m.group(0))

    # you can change this to getctime
    compare_func = lambda w: os.path.getmtime(w)

    best_weight = ''
    if best_weights:
        best_weight = max(best_weights, key=compare_func)

    regular_weights = []
    for weight_file in weight_files:
        m = re.search('.*[0-9]+-regular.pth', weight_file)
        if m:
            regular_weights.append(m.group(0))

    regular_weight = ''
    if regular_weights:
        regular_weight = max(regular_weights, key=compare_func)

    # if we find both -best.pth and -regular.pth
    # return the newest modified one
    if best_weight and regular_weight:
        return max([best_weight, regular_weight], key=compare_func)
    # if we only find -best.pth
    elif best_weight and not regular_weight:
        return best_weight
    # if we only find -regular.pth
    elif not best_weight and regular_weight:
        return regular_weight
    # if we do not found any weight file
    else:
        return ''

def get_model(model_name, input_channels, class_num):

    if model_name == 'unet':
        from models.unet import UNet
        net = UNet(input_channels, class_num)

    elif model_name == 'segnet':
        from models.segnet import SegNet
        net = SegNet(input_channels, class_num)

    elif model_name == 'deeplabv3plus':
        from models.deeplabv3plus import deeplabv3plus
        net = deeplabv3plus(class_num)

    else:
        raise ValueError('network type does not supported')

    return net

#def intersect_and_union(pred_label, label, num_classes, ignore_index):
#    """Calculate intersection and Union.
#    Args:
#        pred_label (ndarray): Prediction segmentation map
#        label (ndarray): Ground truth segmentation map
#        num_classes (int): Number of categories
#        ignore_index (int): Index that will be ignored in evaluation.
#     Returns:
#         ndarray: The intersection of prediction and ground truth histogram
#             on all classes
#         ndarray: The union of prediction and ground truth histogram on all
#             classes
#         ndarray: The prediction histogram on all classes.
#         ndarray: The ground truth histogram on all classes.
#    """
#
#    mask = (label != ignore_index)
#    pred_label = pred_label[mask]
#    label = label[mask]
#
#    intersect = pred_label[pred_label == label]
#    area_intersect, _ = np.histogram(
#        intersect, bins=np.arange(num_classes + 1))
#    area_pred_label, _ = np.histogram(
#        pred_label, bins=np.arange(num_classes + 1))
#    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
#    area_union = area_pred_label + area_label - area_intersect
#
#    return area_intersect, area_union, area_pred_label, area_label
#
#
#def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
#    """Calculate Intersection and Union (IoU)
#    Args:
#        results (list[ndarray]): List of prediction segmentation maps
#        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
#        num_classes (int): Number of categories
#        ignore_index (int): Index that will be ignored in evaluation.
#        nan_to_num (int, optional): If specified, NaN values will be replaced
#            by the numbers defined by the user. Default: None.
#     Returns:
#         float: Overall accuracy on all images.
#         ndarray: Per category accuracy, shape (num_classes, )
#         ndarray: Per category IoU, shape (num_classes, )
#    """
#
#    num_imgs = len(results)
#    assert len(gt_seg_maps) == num_imgs
#    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
#    total_area_union = np.zeros((num_classes, ), dtype=np.float)
#    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
#    total_area_label = np.zeros((num_classes, ), dtype=np.float)
#    for i in range(num_imgs):
#        area_intersect, area_union, area_pred_label, area_label = \
#            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
#                                ignore_index=ignore_index)
#        total_area_intersect += area_intersect
#        total_area_union += area_union
#        total_area_pred_label += area_pred_label
#        total_area_label += area_label
#    all_acc = total_area_intersect.sum() / (total_area_label.sum() + 1e-8)
#    acc = total_area_intersect / (total_area_label + 1e-8)
#    iou = total_area_intersect / (total_area_union + 1e-8)
#    if nan_to_num is not None:
#        return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
#            np.nan_to_num(iou, nan=nan_to_num)
#    return all_acc, acc, iou

def plot_dataset(dataset, out_dir, num=9, class_id=4, ignore_idx=255):
    transforms = dataset.transforms
    for idx in range(num):
        i = random.choice(range(len(dataset)))
        img, label = dataset[i]
        print(np.unique(label))

        label = label
        ignore_mask = label == ignore_idx
        if 0 != class_id:
            label[ignore_mask] = 0
        else:
            label[ignore_mask] = 1

        label[label == class_id] = 255
        label[label != 255] = 0
        label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
        img = np.concatenate((img, label), axis=1)
        cv2.imwrite(os.path.join(out_dir, 'label{}.png'.format(idx)), img)



    dataset.transforms = transforms

def print_eval(class_names, results):
    assert len(class_names) == len(results)
    msg = []
    for cls_idx, (name, res) in enumerate(zip(class_names, results)):
        msg.append('{}(id {}): {:.4f}'.format(name, cls_idx, res))

    #msg = msg.strip()
    #msg = msg[:-1]
    msg = 'Total {} classes '.format(len(results)) + ', '.join(msg)

    print(msg)


def data_loader(args, image_set):

    if args.dataset == 'Camvid':
        dataset = CamVid(
            'data',
            image_set=image_set,
            download=args.download
        )


    elif args.dataset == 'Voc2012':
        dataset = VOC2012Aug(
            'voc_aug',
            image_set=image_set
        )

    else:
        raise ValueError('datset {} not supported'.format(args.dataset))

    if image_set == 'train':
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=dataset.ignore_index),
            transforms.RandomScaleCrop(settings.IMAGE_SIZE),
            transforms.RandomGaussianBlur(),
            transforms.ColorJitter(0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(settings.MEAN, settings.STD),
        ])

    elif image_set == 'val':
        trans = transforms.Compose([
            transforms.RandomScaleCrop(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(settings.MEAN, settings.STD),
        ])

    else:
        raise ValueError('image_set should be one of "train", "val", \
                instead got "{}"'.format(image_set))

    dataset.transforms = trans

    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.b, num_workers=4, shuffle=True, pin_memory=True)

    return data_loader