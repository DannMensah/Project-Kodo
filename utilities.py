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

def resize_and_stack_images_in_dir(directory, h, w, d):
    merged_array = None
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            if not type(merged_array) is np.ndarray:
                img = np.load("{}/{}".format(directory, filename))
                merged_array = resize(img, (h,w,d))
            else:
                img = np.load("{}/{}".format(directory, filename))
                img = resize(img, (h,w,d), mode="reflect")
                merged_array = np.vstack((merged_array, img))
        else:
            continue
    return merged_array

