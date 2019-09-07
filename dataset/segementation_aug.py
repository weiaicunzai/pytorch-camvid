import argparse
import glob
import os

import numpy as np
from PIL import Image


def remove_colormap(filename):
    """Removes the color map from the annotation.
    Args:
        filename: Ground truth annotation filename.
    Returns:
        Annotation without color map.
    """
    return np.array(Image.open(filename))

def save_annotation(annotation, filename):
    """Saves the annotation as png file.
    Args:
        annotation: Segmentation annotation.
        filename: Output filename.
    """
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    pil_image.save(filename)

if __name__ == '__main__':

    parser= argparse.ArgumentParser()
    parser.add_argument('-voc', required=True, help='voc root folder, path_to/VOCdevkit/VOC2012')

    args = parser.parse_args()
    root_folder = args.voc
    aug_folder = os.path.join(root_folder, 'SegmentationClassAug')
    #output folder
    raw_folder = os.path.join(root_folder, 'SegmentationClassAugRaw')

    if not os.path.exists(raw_folder):
        os.mkdir(raw_folder)

    for index, gt_img in enumerate(glob.iglob(os.path.join(aug_folder, '*'))):
        file_name = os.path.basename(gt_img)
        annotation = remove_colormap(gt_img)
        output_path = os.path.join(raw_folder, file_name)
        save_annotation(annotation, output_path)
        print('saving {} file: {}'.format(index, output_path))



