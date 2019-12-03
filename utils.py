
import torch
import numpy as np

#class Metrics:
#    """evaluate predictions, supported metrics: mIOU, F1, presicion,
#    recall
#    """
#
#    def __init__(self, preds, gt, thresh=0.5):
#        """class constructor
#        preds: pytorch tensor object (batchsize, channel, x, y)
#        gt: pytorch tensor object, same shape as preds, only has two distinct
#            value, e.g. (0, 1) or (0, 255)
#        """
#        val = torch.unique(gt).long()
#        assert val.numel() == 2
#        assert preds.size() == gt.size()
#        self.val_max = val[val != 0]
#        self.val_min = val[val == 0]
#
#        self.preds = preds 
#        self.preds[preds > thresh] = self.val_max
#        self.preds[preds <= thresh] = self.val_min
#        self.gt = gt
#        self._batch_size, self._class_num, self._height, self._width = gt.size()
#
#        self.gt = self.gt.long()
#        self.preds = self.preds.long()
#
#        self.confusion_matrix = self._confusion_matrix()
#
#    def _confusion_matrix(self):
#        """generate confusion matrix for later use
#        """
#
#        res = []
#
#        for b_idx in range(self._batch_size):
#            for cls_idx in range(self._class_num):
#                #use dictionary to store confusion matrix
#                #easy to implement, since one pixel can belong
#                #to multiple classes in tableborder segementation
#                confusion_matrix = {}
#
#                p = self.preds[b_idx, cls_idx, :, :]
#                g = self.gt[b_idx, cls_idx, :, :]
#
#                total = 0
#                #predict true to true
#                true_positive_mask = p[g == self.val_max] == self.val_max
#                confusion_matrix['true_positive'] = torch.sum(true_positive_mask).float()
#                total += confusion_matrix['true_positive']
#
#                #predict false to true
#                false_positive_mask = p[g == self.val_min] == self.val_max
#                confusion_matrix['false_positive'] = torch.sum(false_positive_mask).float()
#                total += confusion_matrix['false_positive']
#
#                #predict true to false
#                false_negatigve_mask = p[g == self.val_max] == self.val_min
#                confusion_matrix['false_negative'] = torch.sum(false_negatigve_mask).float()
#                total += confusion_matrix['false_negative']
#
#                #predict false to false
#                true_negative_mask = p[g == self.val_min] == self.val_min
#                confusion_matrix['true_negative'] = torch.sum(true_negative_mask).float()
#                total += confusion_matrix['true_negative']
#
#                assert self._height * self._width == total
#
#                res.append(confusion_matrix)
#
#        return res
#
#    def recall(self):
#        """get recall for each class"""
#
#        res = []
#
#        for cls_idx in range(self._class_num):
#            cms = self.confusion_matrix[cls_idx::self._class_num]
#            assert len(cms) == self._batch_size
#
#            temp = torch.stack([cm['true_positive'].float() / (cm['true_positive'] + 
#                        cm['false_negative'] + 1e-8) for cm in cms], 0)
#
#            res.append(torch.mean(temp))
#
#        return res
#
#    def precision(self):
#        """get precision for each class"""
#
#        res = []
#
#        for cls_idx in range(self._class_num):
#            cms = self.confusion_matrix[cls_idx::self._class_num]
#
#            temp = torch.stack([cm['true_positive'].float() / (cm['true_positive'] + 
#                   cm['false_positive'] + 1e-8) for cm in cms], dim=0)
#            res.append(torch.mean(temp))
#
#        return res
#
#    def iou(self):
#        """get IOU for each class"""
#        res = []
#        for cls_idx in range(self._class_num):
#            cms = self.confusion_matrix[cls_idx::self._class_num]
#
#            iou = torch.stack([cm['true_positive'].float() / 
#                  (cm['true_positive'] + cm['false_positive'] + cm['false_negative']
#                   + 1e-8) for cm in cms], dim=0)
#
#            res.append(torch.mean(iou))
#
#        return res
#
#    def mIOU(self):
#        """get mIOU accross each class"""
#
#        return torch.mean(torch.stack(self.iou(), dim=0))
#
#    def accuracy(self):
#        """get pixel acc for each class"""
#
#        res = []
#
#        for cls_idx in range(self._class_num):
#            cms = self.confusion_matrix[cls_idx::self._class_num]
#
#            acc = [(cm['true_positive'] + cm['true_negative']).float() / 
#                    (cm['true_positive'] + cm['true_negative'] + 
#                    cm['false_negative'] + cm['false_positive'] + 1e-8) for cm in cms]
#
#            acc = torch.stack(acc, dim=0)
#
#            res.append(torch.mean(acc))
#
#        return res
#
#    def F1(self, beta=1):
#
#        precision = self.precision()
#        recall = self.recall()
#
#        precision = torch.stack(precision, dim=0)
#        recall = torch.stack(recall, dim=0)
#
#        precision = torch.mean(precision)
#        recall = torch.mean(recall)
#
#        return (1 + beta ** 2) / (beta ** 2 * (1 / precision + 1 / recall + 1e-8))










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

