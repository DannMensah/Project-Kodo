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
from keras.layers import Dense, Flatten, Conv2D, Dropout, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import backend as K
from sklearn.utils import shuffle
from cv2 import resize

from utilities import (stack_npy_files_in_dir, try_make_dirs,  
                       launch_tensorboard,
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
        self.noise_dim = 100
        self.n_critic = 5
        self.clip_value = 0.01

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
            self.X_control = self.X_control[:, input_channels_mask*2]

        save_path = self.model_path / "data" / data_folder.name
        try_make_dirs(save_path)
        np.save(save_path / "y", self.y)
        np.save(save_path / "X_img", self.X_img)
        np.save(save_path / "X_control", self.X_control)
        with open(save_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def create_encoder(self):
        layers = [
        Conv2D(24, kernel_size=(5, 5), strides=(2, 2), input_shape=(self.img_h, self.img_w, self.img_d)),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.5),

        Conv2D(36, kernel_size=(5, 5), strides=(2, 2)),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.5),

        Conv2D(48, kernel_size=(5, 5), strides=(2, 2)),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.5),

        Conv2D(64, kernel_size=(3, 3)),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dropout(0.5),

        Conv2D(64, kernel_size=(3, 3)),

        Flatten()
        ]

        def encoder(x):
            for layer in layers:
                x = layer(x)
            return x
        return encoder
     
    def create_generator(self):
        noise = Input(shape=(self.noise_dim,))
        input_img = Input(shape=(self.img_h, self.img_w, self.img_d))

        g = self.encoder(input_img)

        g = concatenate([g, noise])
        
        g = Dense(1164)(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = BatchNormalization()(g)
        g = Dropout(0.5)(g)

        g = Dense(100)(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = BatchNormalization()(g)
        g = Dropout(0.5)(g)

        g = Dense(50)(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = BatchNormalization()(g)
        g = Dropout(0.5)(g)

        g = Dense(10)(g)
        g = LeakyReLU(alpha=0.2)(g)

        control_pred = Dense(self.num_control_outputs, activation="tanh")(g)

        return Model([input_img, noise], control_pred)


    def create_discriminator(self):
        input_control = Input(shape=(self.num_control_outputs,))
        input_img = Input(shape=(self.img_h, self.img_w, self.img_d))

        d = self.encoder(input_img)

        d = concatenate([d, input_control])
        
        d = Dense(1164)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        d = Dropout(0.5)(d)

        d = Dense(100)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        d = Dropout(0.5)(d)

        d = Dense(50)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        d = Dropout(0.5)(d)

        d = Dense(10)(d)
        d = LeakyReLU(alpha=0.2)(d)

        valid = Dense(1, activation="linear")(d)

        return Model([input_img, input_control], valid)



    def create_model(self):

        self.num_control_outputs = len(self.info["key_labels"])

        self.encoder = self.create_encoder()

        optimizer = optimizers.Adam(0.0002, 0.5)

        self.discriminator = self.create_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss, 
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.create_generator()
        self.generator.compile(loss=self.wasserstein_loss, 
                               optimizer=optimizer)        

        self.discriminator.summary()
        self.generator.summary()

        # The generator takes noise and the image as inputs and outputs the control
        noise = Input(shape=(self.noise_dim,))
        img = Input(shape=(self.img_h, self.img_w, self.img_d))
        control = self.generator([img, noise])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator([img, control])

        # The combined model  (stacked generator and discriminator) takes
        # noise and the image as input => generates control signal => determines validity 
        self.combined = Model([img, noise], valid)
        self.combined.compile(loss=self.wasserstein_loss, 
            optimizer=optimizer,
            metrics=['accuracy'])


    def train(self, batch_size=50, epochs=100, weights_name="default_weights"):
        X, y = shuffle(self.X, self.y, random_state=0)
        
        split_idx = int(X.shape[0] * 0.8)

        X_train_whole, X_val = X[:split_idx,:,:,:], X[split_idx:,:,:,:]
        y_train_whole, y_val = y[:split_idx,:], y[split_idx:,:]

        weights_path = self.model_path / "weights" / weights_name
        try_make_dirs(weights_path)
        # logs_path = weights_path / "logs"
        # try_make_dirs(logs_path)
        # logs_path_str = str(logs_path.absolute())
        # tb_callback = keras.callbacks.TensorBoard(log_dir=logs_path_str, histogram_freq=0,  
        #   write_graph=True, write_images=True)
        # self.model.compile(loss="mean_squared_error", optimizer=optimizers.adam())
        # launch_tensorboard(logs_path_str) 

        half_batch = int(batch_size / 2)
        iterations = math.ceil(X_train_whole.shape[0] / batch_size)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        valid_half = np.ones((half_batch, 1))
        fake_half = np.zeros((half_batch, 1))

        valid_val = np.ones((split_idx, 1))
        fake_val = np.zeros((split_idx, 1))


        for epoch in range(epochs):
            X_train_whole, y_train_whole = shuffle(X_train_whole, y_train_whole, random_state=0)

            for n_iter in range(iterations):
                batch_start_idx = n_iter * batch_size
                if (n_iter + 1) * batch_size >= X_train_whole.shape[0]:
                    batch_end_idx = X_train_whole.shape[0]
                else:
                    batch_end_idx = (n_iter + 1) * batch_size

                X_train = X_train_whole[batch_start_idx:batch_end_idx,:,:,:]
                y_train = y_train_whole[batch_start_idx:batch_end_idx,:]

                for _ in range(self.n_critic):

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Select a random half batch of controls
                    idx = np.random.randint(0, X_train.shape[0], half_batch)
                    images, controls = X_train[idx], y_train[idx]

                    noise = np.random.normal(0, 1, (half_batch, self.noise_dim))

                    # Generate a half batch of new controls
                    gen_controls = self.generator.predict([images, noise])


                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch([images, controls], valid_half)
                    d_loss_fake = self.discriminator.train_on_batch([images, gen_controls], fake_half)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                    # Clip discriminator weights
                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)


                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))

                idx = np.random.randint(0, X_train.shape[0], batch_size)
                sampled_images = X_train[idx]

                # Train the generator
                g_loss = self.combined.train_on_batch([sampled_images, noise], valid)

                print("Iteration {}/{}".format(n_iter, iterations))

                # Plot the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            noise = np.random.normal(0, 1, (X_val.shape[0], self.noise_dim))
            # Generate a half batch of new controls
            gen_controls = self.generator.predict([X_val, noise])

            d_loss_real_val = self.discriminator.test_on_batch([X_val, y_val], valid_val)
            d_loss_fake_val = self.discriminator.test_on_batch([X_val, gen_controls], valid_fake)
            d_loss_val = 0.5 * np.add(d_loss_fake_val, d_loss_real_val)

            # Train the generator
            g_loss_val = self.combined.test_on_batch([X_val, noise], valid)

            print("Epoch {}/{} D_loss: {} - G_loss: {} - D_loss_val: {} - G_loss_val: {}".format(d_loss, g_loss, d_loss_val, g_loss_val))


        self.generator.save_weights(weights_path / "generator_weights.h5")
        self.discriminator.save_weights(weights_path / "discriminator_weights.h5")
        self.combined.save_weights(weights_path / "combined_weights.h5")

        with open(weights_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)
        
    def get_actions(self, img):
        img = resize(img, (self.img_w, self.img_h)) / 255
        return self.model.predict(np.expand_dims(img, axis=0), batch_size=1)[0]
        
        return prediction

    def get_actions(self, img):
        img = resize(img, (self.img_w, self.img_h)) / 255
        return self.model.predict(np.expand_dims(img, axis=0), batch_size=1)[0]

    def stack_arrays(self, key_events_dir, images_dir, img_update_callback=None):
        images = []
        outputs = []
        sorted_filenames = sorted_alphanumeric(os.listdir(key_events_dir))
        for filename in sorted_filenames:
            if filename.endswith(".npy"):
                frame_idx = filename.split("_")[1].split(".")[0]
                output = np.load(key_events_dir / "key-event_{}.npy".format(frame_idx))
                if self.img_is_dropped(output):
                    continue
                img = np.load(images_dir / "image_{}.npy".format(frame_idx))
                if img_update_callback:
                    img_update_callback(img)
                # Resize, scale and center to [-1, 1]
                img = (resize(img, (self.img_w, self.img_h)) / 127.5) - 1
                images.append(img)
                outputs.append(output)
        X = np.stack(images, axis=0)
        y = np.stack(outputs, axis=0)
        return X, y

    def dropping_function(self, x):
        return (1 / ( 1 + math.exp(-15*(x))))     

    def img_is_dropped(self, actions):
        rand = random.random()
        transformed_magnitude = self.dropping_function(abs(actions[0]) - 0.4)
        braking_magnitude = self.dropping_function((actions[1] + 1)/2 - 0.7)
        if rand < transformed_magnitude:
            return False
        elif rand < braking_magnitude:
            return False
        return True
    
    # Loads all necessary data from the given folder. X's, y's, not info"
    def load_processed_data(self, data_path):
        X = np.load(data_path / "X.npy")
        y = np.load(data_path / "y.npy")
        return (X, y)

    def load_data_into_variables(self, data):
        self.X, self.y, self.info = data
        



