import os
import numpy as np
from utilities import (stack_npy_files_in_dir, try_make_dirs)
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Autopilot"
def process(data_folder):
    y = stack_npy_files_in_dir("{}/key-events".format(data_folder))
    data_folder_name = data_folder.split("/")[-1]
    save_path = "{}/data/{}".format(MODEL_PATH, data_folder_name)
    try_make_dirs(save_path)
    np.save("{}/y".format(save_path), y)
