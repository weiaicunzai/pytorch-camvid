
import numpy as np
from sklearn.metrics import confusion_matrix


class Metrics:

    def __init__(self, class_num, ignore_index=None):
        """compute imou for pytorch segementation task

        Args:
            class_num: predicted class number
            ignore_index: ignore index
        """

        self.class_num = class_num
        self.ignore_index = ignore_index

        # https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
        self._confusion_matrix = np.zeros((self.class_num, self.class_num))

    def add(self, preds : np.array, gts : np.array):
        """update confusion matrix

        Args:
            preds: 1 dimension numpy array, predicted label
            gts: corresponding ground truth for preds, 1 dimension numpy array
        """
        cm = confusion_matrix(gts, preds, labels=range(self.class_num))
        self._confusion_matrix += cm

    def clear(self):
        self._confusion_matrix.fill(0)

    def precision(self, average=True):

        cm = self._confusion_matrix
        precision = np.diag(cm) / (cm.sum(axis=0) + 1e-15)

        if self.ignore_index:
            precision_mask = [i for i in range(self.class_num) if i != self.ignore_index]
            precision = precision[precision_mask]
        if average:
            precision = precision.mean()

        return precision

    def recall(self, average=True):

        cm = self._confusion_matrix
        recall = np.diag(cm) /(cm.sum(axis=1) + 1e-15)

        if self.ignore_index:
            recall_mask = [i for i in range(self.class_num) if i != self.ignore_index]
            recall = recall[recall_mask]
        if average:
            recall = recall.mean()

        return recall

    def iou(self, average=True):

        cm = self._confusion_matrix
        iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-15)
        iou_mask = [i for i in range(self.class_num) if i != self.ignore_index]
        iou = iou[iou_mask]

        if average:
            iou = iou.mean()

        return iou
