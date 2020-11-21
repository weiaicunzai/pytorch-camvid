import os
import tarfile
import shutil
import cv2
import json

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
        self._shapes = {}

        if download:
            download_url(self.url, self._root, self.filename, md5=self.md5)



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

        self._class_names = ['Sky', 'Building', 'Pole', 'Road', 'Pavement',
                            'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian',
                            'Bicyclist', 'Void']
        self.ignore_index = self._class_names.index('Void')

        if not os.path.exists(os.path.join(self._root, 'data.mdb')):
            with tarfile.open(os.path.join(self._root, self._filename), "r") as tar:
                tar.extractall(path=self._root)

            with open(os.path.join(self._root, 'camvid', 'codes.txt')) as f:
                self._codes = [line.strip() for line in f.readlines()]
            # create lmdb dataset
            print('Writing data into lmdb format to acclerate data loading process')
            self._create_lmdb()
            print('Done...')
            shutil.rmtree(os.path.join(self._root, 'camvid'))


        self._env = lmdb.open(self._root, readonly=True, lock=False)

        with self._env.begin(write=False) as txn:
            valid = set(json.loads(txn.get('valid'.encode())))
            images = json.loads(txn.get('images'.encode()))
            self._shapes = json.loads(txn.get('shapes'.encode()))
            self._image_names = []
            for image in images:
                if self._image_set == 'train':
                    if image not in valid:
                        self._image_names.append(image)
                elif self._image_set == 'val':
                    if image in valid:
                        self._image_names.append(image)

                else:
                    raise RuntimeError('wrong image_set value')

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

        masks = [np.zeros(label.shape, dtype='bool') for i in range(len(self._class_names))]
        for cls_id_32 in range(len(self._codes)):
            cls_name_32 = self._codes[cls_id_32]
            cls_name_12 = self._label_IDs[cls_name_32]
            cls_id_12 = self._class_names.index(cls_name_12)
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
            image_shape = self._shapes[image_name]
            label_shape = self._shapes[label_name]
            image = np.ndarray(shape=image_shape, dtype=np.uint8, buffer=image_data)
            label = np.ndarray(shape=label_shape, dtype=np.uint8, buffer=label_data)

        if self.transforms:
                image, label = self.transforms(image, label)

        return image, label


    def _write_lmdb(self, env, img_pathes, flags=-1):

        with env.begin(write=True) as txn:
            for idx, image_path in enumerate(img_pathes):
                image = cv2.imread(image_path, flags).astype(np.uint8)
                if flags == 0:
                    image = self._group_ids(image).astype(np.uint8)
                image_name = os.path.basename(image_path)

                txn.put(image_name.encode(), image.tobytes())
                self._shapes[image_name] = image.shape

                if idx + 1 % 500 == 0:
                    txn.commit()


    def _create_lmdb(self):
        lmdb_map_size = 1 << 40
        env = lmdb.open(self._root, map_size=lmdb_map_size)

        # write images
        image_folder = os.path.join(self._root, 'camvid', 'images')
        label_folder = os.path.join(self._root, 'camvid', 'labels')
        image_names = os.listdir(image_folder)
        image_names = [image_name for image_name in image_names if '.png' in image_name]
        label_names = [name.replace('.', '_P.') for name in image_names]

        image_pathes = [os.path.join(image_folder, name) for name in image_names]
        self._write_lmdb(env, image_pathes, -1)

        label_pathes = [os.path.join(label_folder, name) for name in label_names]
        self._write_lmdb(env, label_pathes, 0)

        # write valid.txt and codes.txt
        with env.begin(write=True) as txn:
            with open(os.path.join(self._root, 'camvid', 'valid.txt')) as f:
                valids = [line.strip() for line in f.readlines()]
                txn.put('valid'.encode(), json.dumps(valids).encode())


            txn.put('shapes'.encode(), json.dumps(self._shapes).encode())
            txn.put('labels'.encode(), json.dumps(label_names).encode())
            txn.put('images'.encode(), json.dumps(image_names).encode())

        env.close()
