import os
import lmdb

def write_data(env, file_pathes):
    with env.begin(write=True) as txn:
        for fidx, fp in enumerate(file_pathes):
            print(fp)
            data_bytes = open(fp, 'rb').read()
            file_name = os.path.basename(fp)
            txn.put(file_name.encode(), data_bytes)

def file_pathes(root, files, ext):
    return [os.path.join(root, fn + ext) for fn in files]

def create_lmdb(out_dir, voc_aug_folder):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    val = [line.strip() for line in open(os.path.join(voc_aug_folder, 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
    trainaug = [line.strip() for line in open(os.path.join(voc_aug_folder, 'ImageSets', 'Segmentation', 'trainaug.txt')).readlines()]

    image_path = os.path.join(voc_aug_folder, 'JPEGImages')
    label_path = os.path.join(voc_aug_folder, 'SegmentationClassAug')


    train_images = file_pathes(image_path, trainaug, '.jpg')
    train_labels = file_pathes(label_path, trainaug, '.png')
    val_images = file_pathes(image_path, val, '.jpg')
    val_labels = file_pathes(label_path, val, '.png')


    #db_size = 1 << 40
    #env = lmdb.open(os.path.join(out_dir, 'train'), map_size=db_size)

    #print('writing training data')
    #write_data(env, train_images + train_labels)

    #env = lmdb.open(os.path.join(out_dir, 'val'), map_size=db_size)
    #print('writing validation data')
    #write_data(env, val_images + val_labels)


if __name__ == '__main__':

    out_dir = 'voc_aug'
    voc_aug_folder = '/data/by/pytorch-camvid/tmp/voc_aug/models-master/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012'
    create_lmdb(out_dir, voc_aug_folder)