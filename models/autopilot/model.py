import os
import json
import threading
from pathlib import Path
from shutil import copyfile

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras import optimizers
from keras import backend as K

from utilities import (stack_npy_files_in_dir, try_make_dirs, resize_and_stack_images_in_dir, 
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
        self.y = stack_npy_files_in_dir(data_folder / "key-events")
        self.y = self.y[:, input_channels_mask]
        save_path = self.model_path / "data" / data_folder.name
        try_make_dirs(save_path)
        np.save(save_path / "y", self.y)
    
        with open(data_folder / "info.json") as info_file:
            data_info = json.load(info_file)

        key_labels = np.asarray(data_info["key_labels"])[input_channels_mask]
        print(key_labels)
        print(input_channels_mask)
        self.info = {
                "key_labels": key_labels.tolist()
                }

        with open(save_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)
        self.X = resize_and_stack_images_in_dir(data_folder / "images", self.img_h, self.img_w, img_update_callback)
        np.save(save_path / "X", self.X)



    def loss(self, y, y_pred):
        return K.sqrt(K.sum(K.square(y_pred-y), axis=-1))

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation="relu", 
                         input_shape=(self.img_h, self.img_w, self.img_d)))
        model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(1164, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.info["key_labels"]), activation="linear"))
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
        img = img_resize_to_int(img, self.img_h, self.img_w)
        return self.model.predict(img, batch_size=1)[0]



