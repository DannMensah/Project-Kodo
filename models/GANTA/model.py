import os
import json
import threading
import math
import random
from pathlib import Path
from shutil import copyfile
import csv

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
        self.clip_value = 0.05

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
        input_img = Input(shape=(self.img_h, self.img_w, self.img_d))
        noise = Input(shape=(self.noise_dim,))

        g = Conv2D(24, kernel_size=(5, 5), strides=(2, 2),
                   input_shape=(self.img_h, self.img_w, self.img_d))(input_img)
        g = LeakyReLU(alpha=0.2)(g)
        g = BatchNormalization()(g)
        g = Dropout(0.5)(g)

        g = Conv2D(36, kernel_size=(5, 5), strides=(2, 2))(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = BatchNormalization()(g)
        g = Dropout(0.5)(g)

        g = Conv2D(48, kernel_size=(5, 5), strides=(2, 2))(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = BatchNormalization()(g)
        g = Dropout(0.5)(g)

        g = Conv2D(64, kernel_size=(3, 3))(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = BatchNormalization()(g)
        g = Dropout(0.5)(g)

        g = Conv2D(64, kernel_size=(3, 3))(g)

        g = Flatten()(g)

        g = concatenate([g, noise])
        
##        g = Dense(1164)(g)
##        g = LeakyReLU(alpha=0.2)(g)
##        g = BatchNormalization()(g)
##        g = Dropout(0.5)(g)

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

        d = Conv2D(24, kernel_size=(5, 5), strides=(2, 2),
                   input_shape=(self.img_h, self.img_w, self.img_d))(input_img)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        d = Dropout(0.5)(d)

        d = Conv2D(36, kernel_size=(5, 5), strides=(2, 2))(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        d = Dropout(0.5)(d)

        d = Conv2D(48, kernel_size=(5, 5), strides=(2, 2))(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        d = Dropout(0.5)(d)

        d = Conv2D(64, kernel_size=(3, 3))(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        d = Dropout(0.5)(d)

        d = Conv2D(64, kernel_size=(3, 3))(d)

        d = Flatten()(d)


        d = concatenate([d, input_control])
        
##        d = Dense(1164)(d)
##        d = LeakyReLU(alpha=0.2)(d)
##        d = BatchNormalization()(d)
##        d = Dropout(0.5)(d)

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

        optimizer_g = optimizers.RMSprop(lr=0.0005)
        optimizer_d = optimizers.RMSprop(lr=0.0005)
        optimizer_conv = optimizers.Adam()


        # The generator takes noise and the image as inputs and outputs the control
        noise = Input(shape=(self.noise_dim,))
        img = Input(shape=(self.img_h, self.img_w, self.img_d))

        #Conv layer pre-trainer
        self.conv_trainer = self.create_generator()
        self.conv_trainer.compile(loss="mean_squared_error",
                                  optimizer=optimizer_conv)

        

        print("\n\n########### DISCRIMINATOR ############")
        self.discriminator = self.create_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss, 
                                   optimizer=optimizer_d)
        self.discriminator.summary()
        

        print("########### GENERATOR ############")
        self.generator = self.create_generator()
        self.discriminator.trainable = False
        self.generator.compile(loss=self.wasserstein_loss, 
                               optimizer=optimizer_g)
        self.generator.summary()

        control = self.generator([img, noise])
        

        print("\n\n########### COMBINED ############")
        valid = self.discriminator([img, control])
        # noise and the image as input => generates control signal => determines validity 
        self.combined = Model([img, noise], valid)
        self.combined.compile(loss=self.wasserstein_loss, 
            optimizer=optimizer_g)
        self.combined.summary()




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

        iterations = math.ceil(X_train_whole.shape[0] / batch_size)

        losses = []
        
        for epoch in range(epochs):
            X_train_whole, y_train_whole = shuffle(X_train_whole, y_train_whole, random_state=0)

            d_losses = []
            g_losses = []

            for n_iter in range(iterations):
                batch_start_idx = n_iter * batch_size
                if (n_iter + 1) * batch_size >= X_train_whole.shape[0]:
                    batch_end_idx = X_train_whole.shape[0]
                else:
                    batch_end_idx = (n_iter + 1) * batch_size

                X_train = X_train_whole[batch_start_idx:batch_end_idx,:,:,:]
                y_train = y_train_whole[batch_start_idx:batch_end_idx,:]

                valid = np.ones((X_train.shape[0], 1), dtype="float32")
                fake = -valid

                half_batch = int(X_train.shape[0] / 2)

                valid_half = np.ones((half_batch, 1), dtype="float32")
                fake_half = -valid_half

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
                    d_loss_real = self.discriminator.train_on_batch([images, controls], fake_half)
                    d_loss_fake = self.discriminator.train_on_batch([images, gen_controls], valid_half)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                    d_losses.append(d_loss)

                    # Clip discriminator weights
                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)


                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (X_train.shape[0], self.noise_dim))

                idx = np.random.randint(0, X_train.shape[0], X_train.shape[0])
                sampled_images = X_train[idx]

                # Train the generator
                g_loss = self.combined.train_on_batch([sampled_images, noise], fake)
                g_losses.append(g_loss)
                print("Iteration {}/{} - D_loss: {:.6f} -  G_loss: {:.6f}".format(n_iter, iterations, d_loss, g_loss))            

            d_val_losses = []
            g_val_losses = []
            val_iterations = math.ceil(X_val.shape[0] / batch_size)
            for n_val_iter in range(val_iterations):
                val_batch_start_idx = n_val_iter * batch_size
                if (n_val_iter + 1) * batch_size >= X_val.shape[0]:
                    val_batch_end_idx = X_val.shape[0]
                else:
                    val_batch_end_idx = (n_val_iter + 1) * batch_size

                X_val_batch = X_val[val_batch_start_idx:val_batch_end_idx,:,:,:]
                y_val_batch = y_val[val_batch_start_idx:val_batch_end_idx,:]

                noise = np.random.normal(0, 1, (X_val_batch.shape[0], self.noise_dim))
                valid = np.ones((X_val_batch.shape[0], 1), dtype="float32")
                fake = -valid

                gen_controls = self.generator.predict([X_val_batch, noise])

                d_loss_real_val = self.discriminator.test_on_batch([X_val_batch, y_val_batch], fake)
                d_loss_fake_val = self.discriminator.test_on_batch([X_val_batch, gen_controls], valid)
                d_loss_val = 0.5 * np.add(d_loss_fake_val, d_loss_real_val)
                g_loss_val = self.combined.test_on_batch([X_val_batch, noise], fake)

                d_val_losses.append(d_loss_val)
                g_val_losses.append(g_loss_val)

            d_loss_val = np.mean(d_val_losses)
            g_loss_val = np.mean(g_val_losses)
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)

            print("Epoch {}/{} AVG_D_loss: {:.6f} - AVG_G_loss: {:.6f} - D_loss_val: {:.6f} - G_loss_val: {:.6f}".format(epoch + 1, epochs, avg_d_loss, avg_g_loss, d_loss_val, g_loss_val))

            losses.append([avg_d_loss, avg_g_loss, d_loss_val, g_loss_val])
            with open(weights_path / "loss.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(losses)
            epoch_suffix = "epoch-{}".format(epoch + 1)
            try_make_dirs(weights_path / epoch_suffix)

            self.generator.save_weights(weights_path / epoch_suffix / "generator_weights.h5")
            self.discriminator.save_weights(weights_path / epoch_suffix / "discriminator_weights.h5")
            self.combined.save_weights(weights_path / epoch_suffix / "combined_weights.h5")
            with open(weights_path / epoch_suffix / "info.json", "w") as info_file:
                json.dump(self.info, info_file)
            

        self.generator.save_weights(weights_path / "generator_weights.h5")
        self.discriminator.save_weights(weights_path / "discriminator_weights.h5")
        self.combined.save_weights(weights_path / "combined_weights.h5")
        with open(weights_path / "info.json", "w") as info_file:
            json.dump(self.info, info_file)
        
    def get_actions(self, img):
        img = (resize(img, (self.img_w, self.img_h)) / 127.5) - 1
        noise = np.zeros((1, self.noise_dim))
        return self.generator.predict([np.expand_dims(img, axis=0), noise])[0]

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

    def load_saved_model(self, saved_model_path):
        self.load_info(saved_model_path / "info.json")
        self.create_model()
        self.generator.load_weights(saved_model_path / "generator_weights.h5")
        self.discriminator.load_weights(saved_model_path / "discriminator_weights.h5")
        self.combined.load_weights(saved_model_path / "combined_weights.h5")
        



