import os
import tarfile
import shutil
import cv2
import glob

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import numpy as np
from conf import settings

class CamVid(Dataset):
    def __init__(self, root, download=False, image_set='train', transforms=None):
        """
        Camvid dataset:https://course.fast.ai/datasets
        or simply wget https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz

        Args:
            data_path: path to dataset folder
            image_set: train datset or validation dataset, 'train', or 'val'
            transforms: data augmentations
        """
        self._image_set = image_set
        self.transforms = transforms
        self._md5 = '2e796d442fe723192014ace89a1515b1'
        self._url = 'https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz'
        self._filename = 'camvid.tgz'
        self._root = root

        if download:
            download_url(self._url, self._root, self._filename, md5=self._md5)

        self._label_IDs = {
            # Sky
            'Sky' : 'Sky',

            # Building
            'Bridge' : 'Building',
            'Building' : 'Building',
            'Wall' : 'Building',
            'Tunnel' : 'Building',
            'Archway' : 'Building',

            # Pole
            'Column_Pole' : 'Pole',
            'TrafficCone' : 'Pole',

            # Road
            'Road' : 'Road',
            'LaneMkgsDriv' : 'Road',
            'LaneMkgsNonDriv' : 'Road',

            # Pavement
            'Sidewalk' : 'Pavement',
            'ParkingBlock' : 'Pavement',
            'RoadShoulder' : 'Pavement',

            # Tree
            'Tree' : 'Tree',
            'VegetationMisc' : 'Tree',

            # SignSymbol
            'SignSymbol' : 'SignSymbol',
            'Misc_Text' : 'SignSymbol',
            'TrafficLight' : 'SignSymbol',

            # Fence
            'Fence' : 'Fence',

            # Car
            'Car' : 'Car',
            'SUVPickupTruck' : 'Car',
            'Truck_Bus' : 'Car',
            'Train' : 'Car',
            'OtherMoving' : 'Car',

            # Pedestrian
            'Pedestrian' : 'Pedestrian',
            'Child' : 'Pedestrian',
            'CartLuggagePram' : 'Pedestrian',
            'Animal' : 'Pedestrian',

            # Bicyclist
            'Bicyclist' : 'Bicyclist',
            'MotorcycleScooter' : 'Bicyclist',

            #Void
            'Void' : 'Void',
        }

        self.class_names = ['Sky', 'Building', 'Pole', 'Road', 'Pavement',
                            'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian',
                            'Bicyclist', 'Void']

        self.ignore_index = self.class_names.index('Void')


        image_fp = os.path.join(self._root, 'camvid', 'images', '*.png')
        if not os.path.exists(os.path.join(self._root, 'camvid')):
            if not os.path.exists(os.path.join(self._root, self._filename)):
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')
            with tarfile.open(os.path.join(self._root, self._filename), "r") as tar:
                tar.extractall(path=self._root)

            with open(os.path.join(self._root, 'camvid', 'codes.txt')) as f:
                self._codes = [line.strip() for line in f.readlines()]


            self._image_names = glob.glob(image_fp)
            print('converting labels from 32 classes to 12 classes and resizing image')
            size = settings.IMAGE_SIZE
            for img in self._image_names:
                image = cv2.imread(img)

                label_path = img.replace('images', 'labels').replace('.', '_P.')
                label = cv2.imread(label_path, 0)
                label = self._group_ids(label).astype(np.uint8)

                cv2.imwrite(label_path, label)

        with open(os.path.join(self._root, 'camvid', 'valid.txt')) as f:
            valids = [line.strip() for line in f.readlines()]

        self._image_names = []
        if image_set == 'train':
            for img in glob.iglob(image_fp):
                if os.path.basename(img) not in valids and '.png' in img:
                    self._image_names.append(img)
        elif image_set in ['val', 'test']:
            self._image_names = [img for img in glob.iglob(image_fp) if os.path.basename(img) in valids]

        else:
            raise RuntimeError('image set should only be train or set')

        self.transforms = transforms
        self.class_names = self.class_names[:-1]
        self.class_num = len(self.class_names)

    def __len__(self):
        return len(self._image_names)

    def _group_ids(self, label):
        """Convert 32 classes camvid dataset to 12 classes by
        grouping the similar class together

        Args:
            label: a 32 clasees gt label
        Returns:
            label: a 12 classes gt label
        """

        masks = [np.zeros(label.shape, dtype='bool') for i in range(len(self.class_names))]
        for cls_id_32 in range(len(self._codes)):
            cls_name_32 = self._codes[cls_id_32]
            cls_name_12 = self._label_IDs[cls_name_32]
            cls_id_12 = self.class_names.index(cls_name_12)
            masks[cls_id_12] += label == cls_id_32


        for cls_id_12, mask in enumerate(masks):
            label[mask] = cls_id_12

        return label

    def __getitem__(self, index):

        image_path = self._image_names[index]
        label_path = image_path.replace('images', 'labels').replace('.', '_P.')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)


        if self.transforms:
                image, label = self.transforms(image, label)

        return image, label
