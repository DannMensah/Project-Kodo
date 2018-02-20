import os
from pathlib import Path
import webbrowser
import threading
import socket

import numpy as np
from skimage.transform import resize

def stack_npy_files_in_dir(directory):
    merged_array = None
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            loaded_arr = np.load(directory / filename)
            if not type(merged_array) is np.ndarray:
                merged_array = loaded_arr  
            else:
                merged_array = np.vstack((merged_array, loaded_arr))
        else:
            continue
    return merged_array


def try_make_dirs(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

def img_resize_to_int(img, h, w, scaled=False):
    img = resize(img, (h,w), mode="reflect")
    if not scaled:
        img = 255 * img
        img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)
    return img


def resize_and_stack_images_in_dir(directory, h, w, img_update_callback=None, scaled=False):
    merged_array = None
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            img = np.load(directory / filename)
            img_update_callback(img)
            img = img_resize_to_int(img, h, w, scaled)
            if not type(merged_array) is np.ndarray:
                merged_array = img            
            else:
                merged_array = np.concatenate((merged_array, img), axis=0)

        else:
            continue
    return merged_array

def launch_tensorboard(log_dir):
    t = threading.Thread(target=run_tensorboard_server, args=([log_dir]))
    t.start()

    url = "http://{}:6006/".format(socket.gethostname())
    webbrowser.open(url, new=0, autoraise=True)
    
def run_tensorboard_server(log_dir):
    print(log_dir)
    os.system("tensorboard --logdir=" + log_dir + " --port 6006")

