import os
from pathlib import Path

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras import optimizers
from keras import backend as K

from utilities import (stack_npy_files_in_dir, try_make_dirs, resize_and_stack_images_in_dir)

class Model:
    def __init__(self):
        self.img_h = 66
        self.img_w = 200
        self.img_d = 3
        self.X = None
        self.y = None
        self.data_name = None
        self.model_path = Path(os.path.dirname(os.path.abspath(__file__)))

    def process(self, data_folder):
        self.y = stack_npy_files_in_dir(self.model_path / "key-events")
        self.data_folder_name = data_folder.split("/")[-1]
        save_path = self.model_path / "data" / self.data_name
        try_make_dirs(save_path)
        np.save(self.save_path / "y")

        self.X = resize_and_stack_images_in_dir(data_folder / "images", self.img_h, self.img_w)
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
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.y.shape[1], activation="tanh"))
        self.model = model

    def load_data(self, data_folder_str):
        self.data_folder_name = data_folder_str.split("/")[-1]
        data_folder = Path(data_folder_str)
        self.y = np.load(data_folder / "y.npy")
        self.X = np.load(data_folder / "X.npy")

    def train(self, batch_size=50, epochs=100):
        self.model.compile(loss=self.loss, optimizer=optimizers.adam())
        self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2)
        try_make_dirs(self.model_path / "weights")
        self.model.save_weights(self.model_path / "weights" / "{}.h5".format(self.data_folder_name))
