import os

import torch
from torch.utils.data import Dataset
from skimage.data import imread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from conf import settings

def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    print(s)
    print(s[0:])
    starts, length = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
    print(type(starts))
    print(type(length))
    starts -= 1
    ends = starts + length
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        print(lo, hi)
        img[lo:hi] = 1
    
    return img.reshape(shape).T
    #fig, axarr = plt.subplots(1, 3, figsize=(15, 40)) 
    #axarr[0].imshow(img)
    #plt.show()

#class ShipDataset

#LABLE_PATH = '/nfs/project/baiyu/dataset/'
train_df = pd.read_csv(os.path.join(settings.LABLE_PATH, 'train_ship_segmentations_v2.csv'))
print(train_df.head())

ImageId = '001a7cba8.jpg'
img_mask = train_df.loc[train_df['ImageId'] == ImageId, 'EncodedPixels'].tolist()
print(img_mask)

all_masks = np.zeros((768, 768))

for mask in img_mask:
    if mask == mask:
        all_masks += rle_decode(mask)

fig, axarr = plt.subplots(1, 1, figsize=(5, 5))
axarr.axis('off')
axarr.imshow(all_masks)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()