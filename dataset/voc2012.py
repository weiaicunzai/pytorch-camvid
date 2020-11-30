

import os
import cv2
import pickle
import numpy as np
import lmdb
from torch.utils.data import Dataset
import torch

class VOC2012Aug(Dataset):

    def __init__(self, root,
                       image_set,
                       transforms=None,
                       ignore_label=255):
        """voc2012 dataset, use augmented voc2012 dataset, resulting:
        training: 10582
        val: 1449
        class_num: 21
        http://home.bharathh.info/pubs/codes/SBD/download.html

        Args:
            data_folder: root folder for voc2012,  path_to/VOCdevkit/VOC2012
            image_set: train dataset or val dataset
            transforms: image and mask transformation
        """

        assert image_set in ['train', 'val']

        self.class_names = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')

        self.class_num = len(self.class_names)
        self.ignore_index = 255

        self.transforms = transforms

        self._root = root
        self._image_set = image_set
        lmdb_path = os.path.join(self._root, self._image_set)
        self._env = lmdb.open(lmdb_path, map_size=1099511627776, readonly=True, lock=False)

        cache_file = os.path.join(lmdb_path, '_cache')

        if os.path.isfile(cache_file):
            self._image_names = pickle.load(open(cache_file, 'rb'))
        else:
            with self._env.begin(write=False) as txn:
                self._image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False) \
                        if '.jpg' in key.decode()]
                pickle.dump(self._image_names, open(cache_file, 'wb'))

    def __getitem__(self, index):

        image_name = self._image_names[index]
        label_name = image_name.replace('.jpg', '.png')
        with self._env.begin(write=False) as txn:
            image_data = txn.get(image_name.encode())
            label_data = txn.get(label_name.encode())

            image = np.frombuffer(image_data, np.uint8)
            label = np.frombuffer(label_data, np.uint8)

            image = cv2.imdecode(image, -1)
            label = cv2.imdecode(label, -1)

        if self.transforms:
                image, label = self.transforms(image, label)

        return image, label

    def __len__(self):
        return len(self._image_names)
