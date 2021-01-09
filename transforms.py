import random
from PIL import Image
import math
import numbers
from collections.abc import Iterable
import warnings
import types
import collections

import cv2
import numpy as np

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance

_cv2_pad_to_str = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT
}

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
        size: expected output size of each edge, can be int or iterable with (w, h)
    """

    def __init__(self, size):

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, Iterable) and len(size) == 2:
            self.size = size
        else:
            raise TypeError('size should be iterable with size 2 or int')

    def __call__(self, img, mask):

        resized_img = cv2.resize(img, self.size)
        resized_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    if len(img.shape) == 3:
        return img[i:i + h, j:j + w, :]
    if len(img.shape) == 2:
        return img[i:i + h, j:j + w]

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given numpy ndarray on all sides with specified padding mode and fill value.
    Args:
        img (numpy ndarray): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        Numpy image: padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(
            type(img)))
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding,
                  collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    #if len(img.shape) == 2:
    #    return cv2.copyMakeBorder(img,
    #                               top=pad_top,
    #                               bottom=pad_bottom,
    #                               left=pad_left,
    #                               right=pad_right,
    #                               borderType=_cv2_pad_to_str[padding_mode],
    #                               value=fill)[:, :, np.newaxis]
    #else:
    return cv2.copyMakeBorder(img,
                              top=pad_top,
                              bottom=pad_bottom,
                              left=pad_left,
                              right=pad_right,
                              borderType=_cv2_pad_to_str[padding_mode],
                              value=fill)

class RandomCrop(object):
    """Crop the given numpy ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, mask):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        if self.padding is not None:
            img = pad(img, self.padding, self.fill, self.padding_mode)
            mask = pad(mask, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            left_pad = int((self.size[1] - img.shape[1]) / 2)
            right_pad = self.size[1] - img.shape[1] - left_pad
            #img = pad(img, (self.size[1] - img.shape[1], 0), self.fill,
            #            self.padding_mode)
            img = pad(img, (left_pad, 0, right_pad, 0), self.fill,
                        self.padding_mode)
            #mask = pad(mask, (self.size[1] - mask.shape[1], 0), self.fill,
            #            self.padding_mode)
            mask = pad(mask, (left_pad, 0, right_pad, 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            top_pad = int((self.size[0] - img.shape[0]) / 2)
            bot_pad = self.size[0] - img.shape[0] - top_pad
            #img = pad(img, (0, self.size[0] - img.shape[0]), self.fill,
            #            self.padding_mode)
            #mask = pad(mask, (0, self.size[0] - mask.shape[0]), self.fill,
            #            self.padding_mode)
            img = pad(img, (0, top_pad, 0, bot_pad), self.fill,
                        self.padding_mode)
            mask = pad(mask, (0, top_pad, 0, bot_pad), self.fill,
                        self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return crop(img, i, j, h, w), crop(mask, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(
            self.size, self.padding)

#class RandomScale:
#    """Randomly scaling an image (from 0.5 to 2.0]), the output image and mask
#    shape will be the same as the input image and mask shape. If the
#    scaled image is larger than the input image, randomly crop the scaled
#    image.If the scaled image is smaller than the input image, pad the scaled
#    image.
#
#    Args:
#        size: expected output size of each edge
#        scale: range of size of the origin size cropped
#        value: value to fill the mask when resizing,
#               should use ignore class index
#    """
#
#    def __init__(self, scale=(0.5, 2.0), value=0):
#
#        if not isinstance(scale, Iterable) and len(scale) == 2:
#            raise TypeError('scale should be iterable with size 2 or int')
#
#        self.value = value
#        self.scale = scale
#
#    def __call__(self, img, mask):
#        oh, ow = img.shape[:2]
#
#        # scale image
#        scale = random.uniform(*self.scale)
#        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
#        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale,
#                          interpolation=cv2.INTER_NEAREST)
#
#        h, w = img.shape[:2]
#
#        # pad image and mask
#        diff_h = max(0, oh - h)
#        diff_w = max(0, ow - w)
#
#        img = cv2.copyMakeBorder(
#            img,
#            diff_h // 2,
#            diff_h - diff_h // 2,
#            diff_w // 2,
#            diff_w - diff_w // 2,
#            cv2.BORDER_CONSTANT,
#            value=[0, 0, 0]
#        )
#        mask = cv2.copyMakeBorder(
#            mask,
#            diff_h // 2,
#            diff_h - diff_h // 2,
#            diff_w // 2,
#            diff_w - diff_w // 2,
#            cv2.BORDER_CONSTANT,
#            value=self.value
#        )
#
#        h, w = img.shape[:2]
#
#        # crop image and mask
#        y1 = random.randint(0, h - oh)
#        x1 = random.randint(0, w - ow)
#        img = img[y1: y1 + oh, x1: x1 + ow]
#        mask = mask[y1: y1 + oh, x1: x1 + ow]
#
#        return img, mask

class RandomRotation:
    """Rotate the image by angle

    Args:
        angle: rotated angle
        value: value used for filling the empty pixel after rotating,
               should use ignore class index

    """

    def __init__(self, p=0.5, angle=10, fill=0):

        if not (isinstance(angle, numbers.Number) and angle > 0):
            raise ValueError('angle must be a positive number.')

        self.angle = angle
        self.value = fill
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        angle = random.uniform(-self.angle, self.angle)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1])
        mask = cv2.warpAffine(
            mask, rot_mat, mask.shape[1::-1],
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.value
        )

        return image, mask

class RandomVerticalFlip:
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
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        return img, mask

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

class RandomGaussianBlur:
    """Blur an image using gaussian blurring.

    Args:
       sigma: Standard deviation of the gaussian kernel.
       Values in the range ``0.0`` (no blur) to ``3.0`` (strong blur) are
       common. Kernel size will automatically be derived from sigma
       p: probability of applying gaussian blur to image

       https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmenters/blur.html#GaussianBlur
    """

    def __init__(self, p=0.5, sigma=(0.0, 3.0)):

        if not isinstance(sigma, Iterable) and len(sigma) == 2:
            raise TypeError('sigma should be iterable with length 2')

        if not sigma[1] >= sigma[0] >= 0:
            raise ValueError(
                'sigma shoule be an iterval of nonegative real number')

        self.sigma = sigma
        self.p = p

    def __call__(self, img, mask):

        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            k_size = self._compute_gaussian_blur_ksize(sigma)
            img = cv2.GaussianBlur(img, (k_size, k_size),
                                   sigmaX=sigma, sigmaY=sigma)

        return img, mask

    @staticmethod
    def _compute_gaussian_blur_ksize(sigma):
        if sigma < 3.0:
            ksize = 3.3 * sigma  # 99% of weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # 97% of weight
        else:
            ksize = 2.6 * sigma  # 95% of weight

        ksize = int(max(ksize, 3))

        # kernel size needs to be an odd number
        if not ksize % 2:
            ksize += 1

        return ksize

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(
            'hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)

def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([i * brightness_factor
                      for i in range(0, 256)]).clip(0, 255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    # ~10ms slower than PIL!
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return np.array(img)

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([(i - 74) * contrast_factor + 74
                      for i in range(0, 256)]).clip(0, 255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, mask):
        return self.lambd(img), mask

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue,
                                     'hue',
                                     center=0,
                                     bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn(
                'Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn(
                'Hue jitter enabled. Will slow down loading immensely.')
        self.p = p

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".
                    format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(
                    name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".
                format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(
                    lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(
                    lambda img: adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, mask):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        if random.random() < self.p:
            return img, mask

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img, mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

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

        mask = torch.from_numpy(mask).long()

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


class RandomScaleCrop:
    """Randomly scaling an image (from 0.5 to 2.0]), the output image and mask
    shape will be the same as the input image and mask shape. If the
    scaled image is larger than the input image, randomly crop the scaled
    image.If the scaled image is smaller than the input image, pad the scaled
    image.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        value: value to fill the mask when resizing,
               should use ignore class index
    """

    def __init__(self, crop_size, scale=(0.5, 2.0), value=0, padding_mode='constant'):

        if not isinstance(scale, Iterable) and len(scale) == 2:
            raise TypeError('scale should be iterable with size 2 or int')

        self.fill = value
        self.scale = scale
        self.crop_size = crop_size
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, mask):

        scale = random.uniform(self.scale[0], self.scale[1])

        crop_size = int(self.crop_size / scale)

        if img.shape[1] < crop_size:
            left_pad = int((crop_size - img.shape[1]) / 2)
            right_pad = crop_size - img.shape[1] - left_pad
            img = pad(img, (left_pad, 0, right_pad, 0), 0,
                        self.padding_mode)
            mask = pad(mask, (left_pad, 0, right_pad, 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if img.shape[0] < crop_size:
            top_pad = int((crop_size - img.shape[0]) / 2)
            bot_pad = crop_size - img.shape[0] - top_pad
            img = pad(img, (0, top_pad, 0, bot_pad), 0,
                        self.padding_mode)
            mask = pad(mask, (0, top_pad, 0, bot_pad), self.fill,
                        self.padding_mode)

        i, j, h, w = self.get_params(img, (crop_size, crop_size))
        img = crop(img, i, j, h, w)
        mask = crop(mask, i, j, h, w)

        img = cv2.resize(img, (self.crop_size, self.crop_size))
        mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)


        return img, mask

#from dataset.camvid import CamVid
#
#
#train_dataset = CamVid(
#        'data',
#        image_set='train'
#)
#
#transform = RandomScaleCrop(473)
#
#train_dataset.transforms = transform
#
#import cv2
#import time
#start = time.time()
#for i in range(100):
#    img, mask = train_dataset[i]
#
#finish = time.time()
#print(100 // (finish - start))
#
#    #cv2.imwrite('test/img{}.png'.format(i), img)
#    #cv2.imwrite('test/mask{}.png'.format(i), mask / mask.max() * 255)
#
#
#

#img = cv2.imread('/data/by/datasets/original/Warwick QU Dataset (Released 2016_07_08)/testB_17.bmp')
##img = cv2.imread('/data/by/pytorch-camvid/data/camvid/images/0001TP_006870.png')
##img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
#print(img.shape)
#mask = np.random.randint(0, 2, size=img.shape[:2])
#print(mask.shape)
#trans = RandomScaleCrop(473)
#trans = RandomRotation(p=1)
#trans = Resize(100)



# Glas transforms
#train_transforms = Compose([
#            #transforms.Resize(settings.IMAGE_SIZE),
#            RandomVerticalFlip(),
#            RandomHorizontalFlip(),
#            RandomRotation(45, fill=11),
#            RandomScaleCrop(473),
#            RandomGaussianBlur(),
#            RandomHorizontalFlip(),
#            ColorJitter(0.4, 0.4),
#            #ToTensor(),
#            #Normalize(settings.MEAN, settings.STD),
#])


#import time
#start = time.time()
#for i in range(10):
#    ig, m = trans(img, mask)
#    #cv2.imwrite('test/{}.jpg'.format(i), ig)
#
#finish = time.time()
#print(finish - start)

class CenterCrop(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        return center_crop(img, self.size), center_crop(mask, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)