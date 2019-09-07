import os
import pickle
import cv2

from torch.utils.data import Dataset
import numpy as np


class TableBorder(Dataset):

    def __init__(self, data_folder, transforms=None):
        super().__init__()

        self.trans = transforms
        self.image_folder = os.path.join(data_folder, 'images')

        #self.mask_data = [
        #   {
        #       img_name : Images_00001.jpeg,
        #       visibel_row_mask : numpy array(0 and 1)
        #       visibel_col_mask : numpy array(0 and 1)
        #   },
        #   {
        #       img_name : Images_00002.jpeg,
        #       visibel_row_mask : numpy array(0 and 1)
        #       visibel_col_mask : numpy array(0 and 1)
        #   },
        #   ...
        # ]
        with open(os.path.join(data_folder, 'mask_data'), 'rb') as f:
            self.mask_data = pickle.load(f)

    def __len__(self):
        return len(self.mask_data)

    def __getitem__(self, index):
        mask_data = self.mask_data[index]
        image = cv2.imread(os.path.join(self.image_folder, mask_data['img_name']))

        visible_row_mask = mask_data['visible_row_mask']
        visible_col_mask = mask_data['visible_col_mask']

        mask = np.dstack((visible_row_mask, visible_col_mask))

        if self.trans:
            image, mask = self.trans(image, mask)

        return image, mask
