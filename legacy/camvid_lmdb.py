import os
import tarfile
import shutil
import cv2
import io
import glob
from itertools import chain

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.datasets import VisionDataset
import numpy as np
import lmdb

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

        self.class_num = len(self.class_names)
        self.ignore_index = self.class_names.index('Void')

        if not os.path.exists(os.path.join(self._root, self._image_set)):
            with tarfile.open(os.path.join(self._root, self._filename), "r") as tar:
                tar.extractall(path=self._root)

            with open(os.path.join(self._root, 'camvid', 'codes.txt')) as f:
                self._codes = [line.strip() for line in f.readlines()]
            print('grouping 32 classes labels into 12 classes....')
            camvid_label_folder = os.path.join(self._root, 'camvid', 'labels', '**', '*.png')
            camvid_images_folder = os.path.join(self._root, 'camvid', 'images', '**', '*.png')
            for label_fp in glob.iglob(camvid_label_folder, recursive=True):
                label = cv2.imread(label_fp, -1)
                label = self._group_ids(label)
                cv2.imwrite(label_fp, label)

            with open(os.path.join(self._root, 'camvid', 'valid.txt')) as f:
                valids = [line.strip() for line in f.readlines()]

            image_pathes = []
            for image_fp in glob.iglob(camvid_images_folder, recursive=True):
                if self._image_set == 'train':
                    if os.path.basename(image_fp) not in valids and 'test.txt' not in image_fp:
                        image_pathes.append(image_fp)
                elif self._image_set == 'val':
                    if os.path.basename(image_fp) in valids:
                        image_pathes.append(image_fp)

                else:
                    raise RuntimeError('image_set should only be one of train val')

            label_pathes = []
            for image_fp in image_pathes:
                basename = os.path.basename(image_fp)
                dirname = os.path.dirname(image_fp)
                sub_folder = os.path.dirname(dirname)
                dirname = os.path.join(sub_folder, 'labels')
                basename = basename.replace('.png', '_P.png')
                label_pathes.append(os.path.join(dirname, basename))

            image_pathes.extend(label_pathes)
            # create lmdb dataset
            print('Writing {} data into lmdb format to acclerate data loading process'.format(self._image_set))
            self._create_lmdb(os.path.join(self._root, self._image_set), image_pathes)
            print('Done...')
            shutil.rmtree(os.path.join(self._root, 'camvid'))



        lmdb_path = os.path.join(self._root, self._image_set)
        self._env = lmdb.open(lmdb_path, map_size=1099511627776, readonly=True, lock=False)

        with self._env.begin(write=False) as txn:
            self._image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False) \
                    if '_P' not in key.decode()]

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

        image_name = self._image_names[index]
        label_name = image_name.replace('.', '_P.')
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

    def _create_lmdb(self, root, image_fps):
        db_size = 1 << 40
        env = lmdb.open(root, map_size=db_size)
        with env.begin(write=True) as txn:
            for fidx, fp in enumerate(image_fps):
                with open(fp, 'rb') as f:
                    img_buff = f.read()

                fn = os.path.basename(fp)
                txn.put(fn.encode(), img_buff)

        env.close()
