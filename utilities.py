import os

import numpy as np

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

