import os
from pathlib import Path
import webbrowser
import threading
import socket
import re

import numpy as np
from cv2 import resize
from operator import itemgetter
from PIL import Image
import colorsys
import random
import copy

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

def launch_tensorboard(log_dir):
    t = threading.Thread(target=run_tensorboard_server, args=([log_dir]))
    t.start()

    #url = "http://{}:6006/".format(socket.gethostname())
    url = "http://localhost:6006/"
    webbrowser.open(url, new=0, autoraise=True)
    
def run_tensorboard_server(log_dir):
    os.system("tensorboard --logdir=" + log_dir + " --port 6006")

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def change_colors(img_arr):
    im = copy.copy(Image.fromarray(img_arr))
    pixdata = im.load()
    colors = im.getcolors(1024)
    sorted_colors = sorted(colors, key=itemgetter(0))
    relevant_colors = sorted_colors[-3:]
    relevant_colors = tuple(col[1] for col in relevant_colors)
    # Clean the background noise, if color != white, then set to black.
    color_map = {}
    hue = random.random()
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            orig_rgba = pixdata[x,y]
            norm_color = tuple(col/255 for col in orig_rgba)
            hls_color = list(colorsys.rgb_to_hls(*norm_color[:3]))
            hls_color[0] = hue
            rgb_color = colorsys.hls_to_rgb(*hls_color)
            rgb_color = tuple(int(col*255) for col in rgb_color)
            pixdata[x,y] = rgb_color
    return np.array(im)


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb
