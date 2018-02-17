import os
import numpy as np
from utilities import (stack_npy_files_in_dir, try_make_dirs, resize_and_stack_images_in_dir)
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Autopilot"


def process(data_folder):
    Y = stack_npy_files_in_dir("{}/key-events".format(data_folder))
    data_folder_name = data_folder.split("/")[-1]
    save_path = "{}/data/{}".format(MODEL_PATH, data_folder_name)
    try_make_dirs(save_path)
    np.save("{}/Y".format(save_path), Y)

    height = 66
    width = 200
    depth = 3
    X = resize_and_stack_images_in_dir("{}/images".format(data_folder), height, width, depth)
    np.save("{}/X".format(save_path), X)
