import os
import json
import threading
import math
import random
from pathlib import Path
from shutil import copyfile

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Dropout, Input, Concatenate
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
        
        self.X_img = None
        self.X_control = None
        self.y = None
        self.img_h = 66
        self.img_w = 200
        self.img_d = 3
        self.data_name = None
        self.model_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.prev_prediction = None

    def process(self, data_folder, input_channels_mask=None, img_update_callback=None):    
        with open(data_folder / "info.json") as info_file:
            data_info = json.load(info_file)
        key_labels = np.asarray(data_info["key_labels"])[input_channels_mask]
        self.info = {
                "key_labels": key_labels.tolist()
                }

        self.X_img, self.X_control, self.y = self.stack_arrays(data_folder / "key-events", 
                                           data_folder / "images", 
                                           img_update_callback)
        if input_channels_mask:
            self.y = self.y[:, input_channels_mask]
            self.X_control = self.X_control[:, input_channels_mask]

        save_path = self.model_path / "data" / data_folder.name
        try_make_dirs(save_path)
        np.save(save_path / "y", self.y)
        np.save(save_path / "X_img", self.X_img)
        np.save(save_path / "X_control", self.X_control)
        with open(save_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)

    def loss(self, y, y_pred):
        return K.sqrt(K.sum(K.square(y_pred-y), axis=-1))

    def create_model(self, dropout_probability=0.5):
        
        input_images = Input(shape=(self.img_h, self.img_w, self.img_d), name="input_images")

        x = Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation="relu")(input_images)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        x = Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        x = Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        x = Flatten()(x)

        x = Dense(1164, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        x = Dense(100, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        x = Dense(50, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        input_controls = Input(shape=(len(self.info["key_labels"]),), name="input_controls")
        x = keras.layers.concatenate([x, input_controls])
        
        x = Dense(10, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_probability)(x)

        main_output = Dense(len(self.info["key_labels"]), activation="softsign", name="main_output")(x)
        
        self.model = Model(inputs=[input_images, input_controls], outputs=main_output)

    def train(self, batch_size=50, epochs=100, weights_name="default_weights"):
        self.X_img, self.X_control, self.y = shuffle(self.X_img, self.X_control, self.y, random_state=0)        
        weights_path = self.model_path / "weights" / weights_name
        try_make_dirs(weights_path)
        logs_path = weights_path / "logs"
        try_make_dirs(logs_path)
        logs_path_str = str(logs_path.absolute())
        tb_callback = keras.callbacks.TensorBoard(log_dir=logs_path_str, histogram_freq=0,  
          write_graph=True, write_images=True)
        self.model.compile(loss=self.loss, optimizer=optimizers.adam())
        launch_tensorboard(logs_path_str) 
        self.model.fit([self.X_img, self.X_control], self.y, batch_size=batch_size, epochs=epochs, shuffle=False, validation_split=0.2,
                       callbacks=[tb_callback])
        self.model.save_weights(weights_path / "weights.h5")
        with open(weights_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)
        
    def get_actions(self, img):
        img = img_resize_to_int(img, self.img_h, self.img_w, scaled=True)
        if type(self.prev_prediction) is not np.ndarray:
            prediction = np.zeros((len(self.info["key_labels"]),))  
        else:
            prediction = self.model.predict([np.expand_dims(img, axis=0), 
                                             np.expand_dims(self.prev_prediction, axis=0)], 
                                             batch_size=1)[0]
        self.prev_prediction = prediction
        return prediction
    def stack_arrays(self, key_events_dir, images_dir, img_update_callback=None):
        images = []
        controls = []
        outputs = []
        sorted_filenames = sorted_alphanumeric(os.listdir(key_events_dir))
        prev_control = None
        for filename in sorted_filenames:
            if filename.endswith(".npy"):
                frame_idx = filename.split("_")[1].split(".")[0]
                output = np.load(key_events_dir / "key-event_{}.npy".format(frame_idx))
                # First image of the dataset, skip to get initial control
                if type(prev_control) is not np.ndarray or self.img_is_dropped(output):
                    prev_control = output
                    continue
                img = np.load(images_dir / "image_{}.npy".format(frame_idx))
                if img_update_callback:
                    img_update_callback(img)
                img = img_resize_to_int(img, self.img_h, self.img_w, scaled=True)
                images.append(img)
                outputs.append(output)
                controls.append(prev_control)
                prev_control = output        
        X_img = np.stack(images, axis=0)
        X_control = np.stack(controls, axis=0)
        y = np.stack(outputs, axis=0)
        return X_img, X_control, y

    def dropping_function(self, x):
        return (1 / ( 1 + math.exp(-15*(x))))     

    def img_is_dropped(self, actions):
        rand = random.random()
        transformed_magnitude = self.dropping_function(abs(actions[0]) - 0.4)
        braking_magnitude = self.dropping_function((actions[1] + 1)/2 - 0.9)
        if rand < transformed_magnitude:
            return False
        elif rand < braking_magnitude:
            return False
 
        return True
    
    # Loads all necessary data from the given folder. X's, y's, not info"
    def load_processed_data(self, data_path):
        X_img = np.load(data_path / "X_img.npy")
        X_control = np.load(data_path / "X_control.npy")
        y = np.load(data_path / "y.npy")
        return (X_img, X_control, y)

    def load_data_into_variables(self, data):
        self.X_img, self.X_control, self.y, self.info = data
        



