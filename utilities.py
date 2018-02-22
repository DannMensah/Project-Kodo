import os
from pathlib import Path
import webbrowser
import threading
import socket
import re

import numpy as np
from skimage.transform import resize

def stack_npy_files_in_dir(directory):
    merged_array = None
    arrays = []
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            loaded_arr = np.load(directory / filename)
            arrays.append(loaded_arr)
        else:
            continue
    merged_array = np.stack(arrays, axis=0)
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
    return img

def launch_tensorboard(log_dir):
    t = threading.Thread(target=run_tensorboard_server, args=([log_dir]))
    t.start()

    url = "http://{}:6006/".format(socket.gethostname())
    webbrowser.open(url, new=0, autoraise=True)
    
def run_tensorboard_server(log_dir):
    os.system("tensorboard --logdir=" + log_dir + " --port 6006")

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
