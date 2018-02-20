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

from utilities import (stack_npy_files_in_dir, try_make_dirs,  
                       img_resize_to_int, launch_tensorboard)
from models.template import KodoModel

class Model(KodoModel):
    def __init__(self):
        super(Model, self).__init__()

        self.img_h = 66
        self.img_w = 200
        self.img_d = 3
        self.data_name = None
        self.model_path = Path(os.path.dirname(os.path.abspath(__file__)))

    def process(self, data_folder, input_channels_mask, img_update_callback=None):    
        with open(data_folder / "info.json") as info_file:
            data_info = json.load(info_file)
        key_labels = np.asarray(data_info["key_labels"])[input_channels_mask]
        self.info = {
                "key_labels": key_labels.tolist()
                }

        self.X, self.y = self.stack_arrays(data_folder / "key-events", 
                                           data_folder / "images", 
                                           img_update_callback)
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
        model.add(BatchNormalization())
        model.add(Dropout(dropout_probability))

        model.add(Dense(len(self.info["key_labels"]), activation="softsign"))

        self.model = model

    def train(self, batch_size=50, epochs=100, weights_name="default_weights"):
        weights_path = self.model_path / "weights" / weights_name
        try_make_dirs(weights_path)
        logs_path = weights_path / "logs"
        try_make_dirs(logs_path)
        logs_path_str = str(logs_path.absolute())
        tb_callback = keras.callbacks.TensorBoard(log_dir=logs_path_str, histogram_freq=0,  
          write_graph=True, write_images=True)
        self.model.compile(loss=self.loss, optimizer=optimizers.adam())
        launch_tensorboard(logs_path_str) 
        self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2,
                       callbacks=[tb_callback])
        self.model.save_weights(weights_path / "weights.h5")
        with open(weights_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)
        
    def get_actions(self, img):
        img = img_resize_to_int(img, self.img_h, self.img_w, scaled=True)
        return self.model.predict(img, batch_size=1)[0]

    def stack_arrays(self, key_events_dir, images_dir, img_update_callback=None):
        images = []
        outputs = []
        for filename in os.listdir(key_events_dir):
            if filename.endswith(".npy"):
                frame_idx = filename.split("_")[1].split(".")[0]
                output = np.load(key_events_dir / "key-event_{}.npy".format(frame_idx))
                if self.img_is_dropped(output):
                    continue
                img  = np.load(images_dir / "image_{}.npy".format(frame_idx))
                img_update_callback(img)
                img = img_resize_to_int(img, self.img_h, self.img_w, scaled=True)
                images.append(img)
                outputs.append(output)
        X = np.stack(images, axis=0)
        y = np.stack(outputs, axis=0)
        return X, y

    def turning_dropping_function(self, x):
        return (1 / ( 1 + math.exp(-10*(x - 0.2))))

    def img_is_dropped(self, actions):
        turning_magnitude = abs(actions[0])
        transformed_magnitude = self.turning_dropping_function(turning_magnitude)
        return random.random() > transformed_magnitude


