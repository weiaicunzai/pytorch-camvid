import random
import math
import numbers

import cv2
import numpy as np

import torch

class Compose:
    """Composes several transforms together.
    Args:
        transforms(list of 'Transform' object): list of transforms to compose
    """    

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):

        for trans in self.transforms:
            img, mask = trans(img, mask)

        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize:
    """Resize an image and an mask to given size
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR: 
    """

    def __init__(self, size):

        self.size = (size, size)

    def __call__(self, img, mask):

        resized_img = cv2.resize(img, self.size)
        resized_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask

class RandomResizedCrop:
    """Randomly crop a rectangle region whose aspect ratio is randomly sampled 
    in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped
    region into a 'size' * 'size' square image.And does the same to mask
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR: 
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'):

        self.methods={
            "area":cv2.INTER_AREA, 
            "nearest":cv2.INTER_NEAREST, 
            "linear" : cv2.INTER_LINEAR, 
            "cubic" : cv2.INTER_CUBIC, 
            "lanczos4" : cv2.INTER_LANCZOS4
        }

        self.size = (size, size)
        self.interpolation = self.methods[interpolation]
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, mask):
        h, w, _ = img.shape

        area = w * h

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            target_ratio = random.uniform(*self.ratio) 

            output_h = int(round(math.sqrt(target_area * target_ratio)))
            output_w = int(round(math.sqrt(target_area / target_ratio))) 

            if random.random() < 0.5:
                output_w, output_h = output_h, output_w 

            if output_w <= w and output_h <= h:
                topleft_x = random.randint(0, w - output_w)
                topleft_y = random.randint(0, h - output_h)
                break

        if output_w > w or output_h > h:
            output_w = min(w, h)
            output_h = output_w
            topleft_x = random.randint(0, w - output_w) 
            topleft_y = random.randint(0, h - output_w)

        cropped_img = img[topleft_y : topleft_y + output_h, topleft_x : topleft_x + output_w]
        cropped_mask = mask[topleft_y : topleft_y + output_h, topleft_x : topleft_x + output_w]

        resized_img = cv2.resize(cropped_img, self.size, interpolation=self.interpolation)
        resized_mask = cv2.resize(cropped_mask, self.size, interpolation=self.interpolation)

        return resized_img, resized_mask

class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.
    and does the same to mask

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        
        return img, mask

class ColorJitter:

    """Randomly change the brightness, contrast and saturation of an image
    Args:
        brightness: (float or tuple of float(min, max)): how much to jitter
            brightness, brightness_factor is choosen uniformly from[max(0, 1-brightness),
            1 + brightness] or the given [min, max], Should be non negative numbe
        contrast: same as brightness
        saturation: same as birghtness
        hue: same as brightness
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast)
        self.saturation = self._check_input(saturation)
        self.hue = self._check_input(hue)

    def _check_input(self, value):

        if isinstance(value, numbers.Number):
            assert value >= 0, 'value should be non negative'
            value = [max(0, 1 - value), 1 + value]
        
        elif isinstance(value, (list, tuple)):
            assert len(value) == 2, 'brightness should be a tuple/list with 2 elements'
            assert 0 <= value[0] <= value[1], 'max should be larger than or equal to min,\
            and both larger than 0'

        else:
            raise TypeError('need to pass int, float, list or tuple, instead got{}'.format(type(value).__name__))

        return value

    def __call__(self, img, mask):
        """
        Args:
            img to be jittered
        Returns:
            jittered img
        """

        img_dtype = img.dtype
        h_factor = random.uniform(*self.hue)
        b_factor = random.uniform(*self.brightness)
        s_factor = random.uniform(*self.saturation)
        c_factor = random.uniform(*self.contrast)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img.astype('float32')

        #h
        img[:, :, 0] *= h_factor
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        #s
        img[:, :, 1] *= s_factor
        img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)

        #v
        img[:, :, 2] *= b_factor
        img[:, :, 2] = np.clip(img[:, :, 2], 0, 255)

        img = img.astype(img_dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        #c
        img = img * c_factor
        img = img.astype(img_dtype)
        img = np.clip(img, 0, 255)

        return img, mask

class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch 
    float tensor (c, h, w) ranged from 0 to 1, and convert mask to torch tensor
    """

    def __call__(self, img, mask):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]

        Returns:
            a pytorch tensor
        """
        #convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = mask.float()

        return img, mask

class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    and does nothing to mask tensor
    
    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img, mask):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """        
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        if not self.inplace:
            img = img.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img, mask