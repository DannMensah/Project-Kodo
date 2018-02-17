import os

import numpy as np
from skimage.transform import resize

def stack_npy_files_in_dir(directory):
    merged_array = None
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            if not type(merged_array) is np.ndarray:
                merged_array = np.load("{}/{}".format(directory, filename))
            else:
                merged_array = np.vstack((merged_array, np.load("{}/{}".format(directory, filename))))
        else:
            continue
    return merged_array


def try_make_dirs(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

def img_resize_to_int(img, h, w):
    img = resize(img, (h,w), mode="reflect")
    img = 255 * img
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)
    return img


def resize_and_stack_images_in_dir(directory, h, w):
    merged_array = None
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            img = np.load("{}/{}".format(directory, filename))
            if not type(merged_array) is np.ndarray:
                merged_array = img_resize_to_int(img, h, w)
            else:
                img = img_resize_to_int(img, h, w)
                merged_array = np.concatenate((merged_array, img), axis=0)
        else:
            continue
    return merged_array
