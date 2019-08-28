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

def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset
    
    Returns:
        return: mean and std of this dataset
    """

    mean_r = 0
    mean_g = 0
    mean_b = 0

    for img, _ in dataset:
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std

#dataset = TableBorder("/home/baiyu/Downloads/web_crawler")
#print(len(dataset))
###
#image, mask = dataset[33]
#
##for i in dataset:
##    img = i[0]
##    print(img.shape)
#
#print(compute_mean_and_std(dataset))
#
##
##print(mask.shape)
#mask = cv2.resize(mask, (512, 512))
#trans = RandomResizedCrop(512)
#image, mask = trans(image, mask)
#
#trans = RandomHorizontalFlip()
#image, mask = trans(image, mask)
#
#trans = ColorJitter()
#image, mask = trans(image, mask)
##
##print(np.max(mask))
##print(mask.dtype)
##
#cv2.imshow("mask1", mask[: , :, 0] * 255)
#cv2.imshow("mask2", mask[: , :, 1] * 255)
#cv2.imshow("image", image)
#cv2.waitKey(0)
##print(mask.shape)
##print(mask.shape)
##a = mask.shape[0] * mask.shape[1] * mask.shape[2]
##print(np.sum(mask) / a)
##
##cv2.imshow("ff", image)
##print(image.shape)
##print(mask.shape)
##cv2.imshow("hh", mask[:, :, 2] * 255)
##cv2.waitKey(0)
##
##
##
##
###image_path = "/home/baiyu/Downloads/web_crawler/images/Image_00002.jpeg"
##
###img = io.imread(image_path)
###img = resize(img, (512, 512))
###io.imshow(img)
###plt.show()