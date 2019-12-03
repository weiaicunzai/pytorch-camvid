
import numpy as np
from sklearn.metrics import confusion_matrix


class Metric:

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

    def add(preds : np.array, gts : np.array):
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
        
        precision = self._confusion_matrix.sum(axis=0)
        if self.ignore_index:
            precision = [precision[i] for i in range(self.class_num) if i != self.ignore_index]
        if self.average:
            precision = precision.mean()

        return precision
    def recall(self, average=True):
        recall = self._confusion_matrix.sum(axis=1)

        if self.ignore_index:
            recall = [recall[i] for i in range(self.class_num) if i != self.class_num]
        if self.average:
            recall = recall.eman()

        return recall

    def iou(self, average=True):

        recall = self.recall(average=False)
        precision = self.precision(average=False)
        cm = self._confusion_matrix
        iou = cm.diag() / (recall + precision + cm.diag() + 1e-8)

        if average:
            iou = iou.mean()

        return iou



