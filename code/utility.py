##############################################################################
###                              utility.py                                ###
##############################################################################

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label


#########################  load and transform data  ###########################

def getResizeImage(path, height=256, width=256, channels=3):
    '''
    Read and resize images
    '''
    # get all folder ids
    ids = next(os.walk(path))[1]

    # initialize vector
    imgs = np.zeros((len(ids), height, width, channels), dtype=np.dtype(float).type)
    size_ori = []

    # read images
    print('Getting and resizing images ... ', flush=True)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        tmp_path = os.path.join(path, id_, 'images', id_+'.png')
        tmp_img = imread(tmp_path)[:,:,:3]  # RGB channels
        size_ori.append(tmp_img.shape[:2])  # (height, width)
        tmp_img = resize(tmp_img, output_shape=(height, width), mode='constant', preserve_range=False)
        imgs[n] = tmp_img
    
    return imgs, ids, size_ori


def getResizeMask(path, height=256, width=256, threshold=0.5):
    '''
    Read and resize mask
    '''
    # get all folder ids
    ids = next(os.walk(path))[1]

    # initialize vector
    masks = np.zeros((len(ids), height, width, 1), dtype=np.uint8)

    # read images
    print('Getting and resizing masks ... ', flush=True)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        tmp_mask = np.zeros((height, width, 1), dtype=np.bool)
        tmp_path = os.path.join(path, id_, 'masks')

        for mask_file in next(os.walk(tmp_path))[2]:
            mask_ = imread(os.path.join(tmp_path, mask_file))
            mask_ = resize(mask_, output_shape=(height, width), mode='constant', preserve_range=False) > threshold # original is > 0
            mask_ = np.expand_dims(mask_, axis=-1)
            tmp_mask = np.maximum(tmp_mask, mask_)

        masks[n] = tmp_mask
    
    return masks


def restoreImage(imgs, size_ori, threshold):
    '''
    Resize images to original size
    @param imgs: predicted images
    @param size_ori: original image sizes before transformation
    @param threshold: cutoff of a pixal to be 1 after resized
    @return restored images
    '''
    print('Restoring images to original sizes ... ', flush=True)
    imgs_ori = []
    for i in range(len(imgs)):
        tmp_imgs_ = resize(np.squeeze(imgs[i]), output_shape=(size_ori[i][0], size_ori[i][1]), mode='constant', preserve_range=False)
        imgs_ori.append(tmp_imgs_ > threshold)
    
    return imgs_ori


#########################  prepare for submission file  ###########################

def _rle_encoding(x):
    '''
    Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    '''
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): 
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def _prob_to_rles(y):
    '''
    Detect all objects in the prediction
    '''
    lab_img = label(y)
    for i in range(1, lab_img.max() + 1):
        yield _rle_encoding(lab_img == i)


def convert2Sub(y_pred, ids):
    '''
    Write predection vectors to submission file
    '''
    sub_ids = []
    rles = []

    # iterate over the test IDs, generate run-length encodings
    print('Preparing submission data for test images ... ', flush=True)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        rle_ = list(_prob_to_rles(y_pred[n]))
        rles.extend(rle_)
        sub_ids.extend([id_] * len(rle_))

    # create submission dataframe
    submission = pd.DataFrame()
    submission['ImageId'] = sub_ids
    submission['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(_) for _ in x))

    return submission