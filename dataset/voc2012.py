

import os 
import cv2
import numpy as np
from torch.utils.data import Dataset

class VOC2012(Dataset):

    def __init__(self, data_folder,
                       dataset,
                       transforms=None,
                       ignore_label=255,
                       class_num=21):
        """voc2012 dataset, use augmented voc2012 dataset, resulting:
        training: 10582
        val: 1449
        class_num: 21
        http://home.bharathh.info/pubs/codes/SBD/download.html

        Args:
            data_folder: root folder for voc2012,  path_to/VOCdevkit/VOC2012
            dataset: train dataset or val dataset
            transforms: image and mask transformation
        """

        assert dataset in ['train', 'val']

        self.ignore_lable = ignore_label
        self.class_num = class_num
        self._trans = transforms
        self._img_folder = os.path.join(data_folder, 'JPEGImages')
        self._seg_folder = os.path.join(data_folder, 'SegmentationClassAugRaw')
        self._datasplit_folder = os.path.join(data_folder, 'ImageSets', 'Segmentation')
        self._filenames = self._get_filenames(dataset)

    def __getitem__(self, index):

        img_path = os.path.join(self._img_folder, self._filenames[index] + '.jpg')
        seg_path = os.path.join(self._seg_folder, self._filenames[index] + '.png')

        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path, 0)

        #label = np.zeros((*seg.shape, self.class_num), dtype=np.uint8)

        #not necessarily a one-hot encoding, since 255 pixel value is igored
        #for cls_idx in range(self.class_num):
        #    label[:, :, cls_idx][seg == cls_idx] = 1
        #    label[:, :, cls_idx][seg != cls_idx] = 0

        return img, seg

    def __len__(self):
        return len(self._filenames)

    def _get_filenames(self, dataset):

        res = []
        dataset = 'trainaug.txt' if dataset == 'train' else 'val.txt'

        with open(os.path.join(self._datasplit_folder, dataset)) as f:
                for line in f:
                    res.append(line.strip())

        return res





