import os
import cv2

from torch.utils.data import Dataset
import numpy as np

class CamVid(Dataset):
    def __init__(self, data_path, class_num, data_type='train', transforms=None):

        self.class_num = class_num
        self.data_type = data_type
        self.data_path = data_path
        self.transforms = transforms

        valid_names = []
        with open(os.path.join(self.data_path, 'valid.txt')) as f:
            for line in f.readlines():
                valid_names.append(line.strip())

        valid = np.loadtxt(os.path.join(self.data_path, 'valid.txt'), dtype=str)
        codes = np.loadtxt(os.path.join(self.data_path, 'codes.txt'), dtype=str)
        self.ignore_index = list(codes).index('Void')

        images = set(os.listdir(os.path.join(self.data_path, 'images')))
        if self.data_type == 'train':
            self.image_names = list(images - set(valid))
        elif self.data_type == 'val':
            self.image_names = list(valid)
        else:
            raise ValueError('data_type should be one of train, val')


    def __getitem__(self, index):
        
        image_name = self.image_names[index]
        image_path = os.path.join(self.data_path, 'images', image_name)
        label_path = os.path.join(self.data_path, 'labels', image_name.replace('.', '_P.'))

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)

        if self.transforms:
            image, label = self.transforms(iamge, label)

        return image, label

    def __len__(self):
        return len(self.image_names)