import os
import json
import threading
import math
import random
from pathlib import Path
from shutil import copyfile

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import backend as K
from sklearn.utils import shuffle

from utilities import (stack_npy_files_in_dir, try_make_dirs,  
                       img_resize_to_int, launch_tensorboard,
                       sorted_alphanumeric)
from models.template import KodoTemplate 

class KodoModel(KodoTemplate):
    def __init__(self):
        super(KodoModel, self).__init__()

        self.img_h = 66
        self.img_w = 200
        self.img_d = 3
        self.data_name = None
        self.model_path = Path(os.path.dirname(os.path.abspath(__file__)))

    def process(self, data_folder, input_channels_mask=None, img_update_callback=None):    
        with open(data_folder / "info.json") as info_file:
            data_info = json.load(info_file)
        key_labels = np.asarray(data_info["key_labels"])[input_channels_mask]
        self.info = {
                "key_labels": key_labels.tolist()
                }

        self.X, self.y = self.stack_arrays(data_folder / "key-events", 
                                           data_folder / "images", 
                                           img_update_callback)
        if input_channels_mask:
            self.y = self.y[:, input_channels_mask]

        save_path = self.model_path / "data" / data_folder.name
        try_make_dirs(save_path)
        np.save(save_path / "y", self.y)
        np.save(save_path / "X", self.X)
        with open(save_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)

    def loss(self, y, y_pred):
        return K.sqrt(K.sum(K.square(y_pred-y), axis=-1))

    def create_model(self, dropout_probability=0.5):
        model = Sequential()

        model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation="relu", 
                         input_shape=(self.img_h, self.img_w, self.img_d)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Flatten())

        model.add(Dense(1164, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Dense(100, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Dense(50, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Dense(10, activation="relu"))

        model.add(Dense(len(self.info["key_labels"]), activation="softmax"))

        self.model = model

    def train(self, batch_size=50, epochs=100, weights_name="default_weights"):
        self.X, self.y = shuffle(self.X, self.y, random_state=0)        
        weights_path = self.model_path / "weights" / weights_name
        try_make_dirs(weights_path)
        logs_path = weights_path / "logs"
        try_make_dirs(logs_path)
        logs_path_str = str(logs_path.absolute())
        tb_callback = keras.callbacks.TensorBoard(log_dir=logs_path_str, histogram_freq=0,  
          write_graph=True, write_images=True)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.adam(), metrics=["accuracy"])
        launch_tensorboard(logs_path_str) 
        self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2,
                       callbacks=[tb_callback])
        self.model.save_weights(weights_path / "weights.h5")
        with open(weights_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)
        
    def get_actions(self, img):
        img = img_resize_to_int(img, self.img_h, self.img_w, scaled=True)
        actions = self.model.predict(np.expand_dims(img, axis=0), batch_size=1)[0]
        output_actions = [0] * actions.shape[0]
        max_action_idx = np.argmax(actions)
        print(max_action_idx)
        output_actions[max_action_idx] = 1
        return output_actions

    def stack_arrays(self, key_events_dir, images_dir, img_update_callback=None):
        images = []
        outputs = []
        n_directions = None
        sorted_filenames = sorted_alphanumeric(os.listdir(key_events_dir))
        for filename in sorted_filenames:
            if filename.endswith(".npy"):
                frame_idx = filename.split("_")[1].split(".")[0]
                output = np.load(key_events_dir / "key-event_{}.npy".format(frame_idx))
                if type(n_directions) is not np.ndarray:
                    n_directions = np.zeros(output.shape[0])
                for idx, out in enumerate(output):
                    if out == 1:
                        n_directions[idx] += 1
                if self.img_is_dropped(output):
                    continue
                img = np.load(images_dir / "image_{}.npy".format(frame_idx))
                if img_update_callback:
                    img_update_callback(img)
                img = img_resize_to_int(img, self.img_h, self.img_w, scaled=True)
                images.append(img)
                outputs.append(output)
        images, outputs = self.even_directions(images, outputs, n_directions)
        X = np.stack(images, axis=0)
        y = np.stack(outputs, axis=0)
        return X, y

    def even_directions(self, images, outputs, n_directions):
        images, outputs = shuffle(images, outputs, random_state=0)
        n_keep = np.min(n_directions[np.nonzero(n_directions)])        
        n_kept = [0] * n_directions.shape[0]
        drop_list = []
        for idx, output in enumerate(outputs):
            keep = True
            for var_idx, out_val in enumerate(output):
                if out_val == 1 and n_kept[var_idx] >= n_keep:
                    keep = False
            if not keep:
                drop_list.append(idx)
            else:
                for idx, out_val in enumerate(output):
                    n_kept[idx] += out_val
        images = np.delete(images, drop_list, 0)
        outputs = np.delete(outputs, drop_list, 0)
        return (images, outputs)
        
    def img_is_dropped(self, actions):
        return sum(actions) == 0
    
    # Loads all necessary data from the given folder. X's, y's, not info"
    def load_processed_data(self, data_path):
        X = np.load(data_path / "X.npy")
        y = np.load(data_path / "y.npy")
        return (X, y)

    def load_data_into_variables(self, data):
        self.X, self.y, self.info = data
        

