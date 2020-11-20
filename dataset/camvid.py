import os
import cv2

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.datasets import VisionDataset
import numpy as np

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

        self.data_type = image_set
        self.data_path = root
        self.transforms = transforms
        self.md5 = '2e796d442fe723192014ace89a1515b1'
        self.url = 'https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz'
        self.filename = 'camvid.tgz'

        if download:
            download_url(self.url, self.data_path, self.filename, md5=self.md5)
            with tarfile.open(os.path.join(self.data_path, self.filename), "r") as tar:
                tar.extractall(path=self.data_path)

        camvid_dir = os.path.join(self.root, 'camvid')
        if not os.path.isdir(camvid_dir):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.label_IDs = {

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

        valid_names = []
        with open(os.path.join(self.data_path, 'valid.txt')) as f:
            for line in f.readlines():
                valid_names.append(line.strip())

        valid = np.loadtxt(os.path.join(self.data_path, 'valid.txt'), dtype=str)
        self.codes = np.loadtxt(os.path.join(self.data_path, 'codes.txt'), dtype=str)
        self.ignore_index = self.class_names.index('Void')

        images = os.listdir(os.path.join(self.data_path, 'images'))
        images = {image for image in images if image.endswith('.png')}

        if self.data_type == 'train':
            self.image_names = list(images - set(valid))
        elif self.data_type == 'val':
            self.image_names = list(valid)
        else:
            raise ValueError('data_type should be one of train, val')

        self.class_num = len(self.class_names)

    #def download_extract(self, url: str, root: str, filename: str, md5: str) -> None:


    def __getitem__(self, index):

        image_name = self.image_names[index]
        image_path = os.path.join(self.data_path, 'images', image_name)
        label_path = os.path.join(self.data_path, 'labels', image_name.replace('.', '_P.'))

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)

        label = self._group_ids(label)

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self):
        return len(self.image_names)

    def _group_ids(self, label):
        """Convert 32 classes camvid dataset to 12 classes by
        grouping the similar class together

        Args:
            label: a 32 clasees gt label
        Returns:
            label: a 12 classes gt label
        """

        masks = [np.zeros(label.shape, dtype='bool') for i in range(len(self.class_names))]
        for cls_id_32 in range(len(self.codes)):
            cls_name_32 = self.codes[cls_id_32]
            cls_name_12 = self.label_IDs[cls_name_32]
            cls_id_12 = self.class_names.index(cls_name_12)
            masks[cls_id_12] += label == cls_id_32


        for cls_id_12, mask in enumerate(masks):
            label[mask] = cls_id_12

        return label

data = CamVid('data', download=True)