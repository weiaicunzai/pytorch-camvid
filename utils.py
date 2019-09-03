
import torch

class Metrics:
    """evaluate predictions, supported metrics: mIOU, F1, presicion,
    recall
    """

    def __init__(self, preds, gt, thresh=0.5):
        """class constructor
        preds: pytorch tensor object (batchsize, channel, x, y)
        gt: pytorch tensor object, same shape as preds, only has two distinct
            value, e.g. (0, 1) or (0, 255)
        """
        val = torch.unique(gt).long()
        assert val.numel() == 2
        assert preds.size() == gt.size()
        self.val_max = val[val != 0]
        self.val_min = val[val == 0]

        self.preds = preds 
        self.preds[preds > thresh] = self.val_max
        self.preds[preds <= thresh] = self.val_min
        self.gt = gt
        self._batch_size, self._class_num, self._height, self._width = gt.size()

        self.gt = self.gt.long()
        self.preds = self.preds.long()

        self.confusion_matrix = self._confusion_matrix()

    def _confusion_matrix(self):
        """generate confusion matrix for later use
        """

        res = []

        for b_idx in range(self._batch_size):
            for cls_idx in range(self._class_num):
                #use dictionary to store confusion matrix
                #easy to implement, since one pixel can belong
                #to multiple classes in tableborder segementation
                confusion_matrix = {}

                p = self.preds[b_idx, cls_idx, :, :]
                g = self.gt[b_idx, cls_idx, :, :]

                total = 0
                #predict true to true
                true_positive_mask = p[g == self.val_max] == self.val_max
                confusion_matrix['true_positive'] = torch.sum(true_positive_mask).float()
                total += confusion_matrix['true_positive']

                #predict false to true
                false_positive_mask = p[g == self.val_min] == self.val_max
                confusion_matrix['false_positive'] = torch.sum(false_positive_mask).float()
                total += confusion_matrix['false_positive']

                #predict true to false
                false_negatigve_mask = p[g == self.val_max] == self.val_min
                confusion_matrix['false_negative'] = torch.sum(false_negatigve_mask).float()
                total += confusion_matrix['false_negative']

                #predict false to false
                true_negative_mask = p[g == self.val_min] == self.val_min
                confusion_matrix['true_negative'] = torch.sum(true_negative_mask).float()
                total += confusion_matrix['true_negative']

                assert self._height * self._width == total

                res.append(confusion_matrix)

        return res

    def recall(self):
        """get recall for each class"""

        res = []

        for cls_idx in range(self._class_num):
            cms = self.confusion_matrix[cls_idx::self._class_num]
            assert len(cms) == self._batch_size

            temp = torch.stack([cm['true_positive'].float() / (cm['true_positive'] + 
                        cm['false_negative'] + 1e-8) for cm in cms], 0)

            res.append(torch.mean(temp))

        return res

    def precision(self):
        """get precision for each class"""

        res = []

        for cls_idx in range(self._class_num):
            cms = self.confusion_matrix[cls_idx::self._class_num]

            temp = torch.stack([cm['true_positive'].float() / (cm['true_positive'] + 
                   cm['false_positive'] + 1e-8) for cm in cms], dim=0)
            res.append(torch.mean(temp))

        return res

    def iou(self):
        """get IOU for each class"""
        res = []
        for cls_idx in range(self._class_num):
            cms = self.confusion_matrix[cls_idx::self._class_num]

            iou = torch.stack([cm['true_positive'].float() / 
                  (cm['true_positive'] + cm['false_positive'] + cm['false_negative']
                   + 1e-8) for cm in cms], dim=0)

            res.append(torch.mean(iou))

        return res

    def mIOU(self):
        """get mIOU accross each class"""

        return torch.mean(torch.stack(self.iou(), dim=0))

    def accuracy(self):
        """get pixel acc for each class"""

        res = []

        for cls_idx in range(self._class_num):
            cms = self.confusion_matrix[cls_idx::self._class_num]

            acc = [(cm['true_positive'] + cm['true_negative']).float() / 
                    (cm['true_positive'] + cm['true_negative'] + 
                    cm['false_negative'] + cm['false_positive'] + 1e-8) for cm in cms]

            acc = torch.stack(acc, dim=0)

            res.append(torch.mean(acc))

        return res

    def F1(self, beta=1):

        precision = self.precision()
        recall = self.recall()

        precision = torch.stack(precision, dim=0)
        recall = torch.stack(recall, dim=0)

        precision = torch.mean(precision)
        recall = torch.mean(recall)

        return (1 + beta ** 2) / (beta ** 2 * (1 / precision + 1 / recall + 1e-8))








#import cv2

#gt1 = cv2.imread("mask1.jpg", 0)
#gt2 = cv2.imread("mask2.jpg", 0)

#gt1[gt1 > 128] = 255
#gt1[gt1 != 255] = 0
##cv2.imwrite("test1.jpeg", test1)
##cv2.imshow("gt2", gt2)
#gt2[gt2 > 128] = 255
#gt2[gt2 != 255] = 0

#gt1 = torch.from_numpy(gt1).float()
#gt2 = torch.from_numpy(gt2).float()



#pred1 = cv2.imread("preds1.jpg", 0)
#pred2 = cv2.imread("preds2.jpg", 0)

##cv2.imshow("pred1", pred1)
##cv2.imshow("pred2", pred2)

#pred1 = torch.from_numpy(pred1).float()
#pred2 = torch.from_numpy(pred2).float()


#pred = torch.stack([pred1, pred2], dim=0)
#pred = pred.unsqueeze(0)


#gt = torch.stack([gt1, gt2], dim=0)
#gt = gt.unsqueeze(0)

#pred = torch.cat([pred, pred], dim=0)
#gt = torch.cat([gt, gt], dim=0)

#pred = torch.cat([pred, pred], dim=0)
#gt = torch.cat([gt, gt], dim=0)

#print(pred.size())
#print(gt.size())

#metric = Metrics(pred.cuda(), gt.cuda())

#res = metric.confusion_matrix

#recall = metric.recall()
#precision = metric.precision()
#ious = metric.IOU()
#mIOU = metric.mIOU()
#accuracy = metric.accuracy()
#F1 = metric.F1()

#print(recall)
#print(precision)
#print(ious)
#print(mIOU)
#print(accuracy)

#print("F1")
#print(F1)
##print(torch.mean(F1))
##print((F1[0] + F1[1]) / 2)





