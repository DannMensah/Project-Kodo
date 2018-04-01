import os
import json
import threading
import math
import random
from pathlib import Path
from shutil import copyfile
import csv
from functools import partial


import numpy as np
import keras
from keras.models import Model
from keras.layers.merge import _Merge
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

class RandomWeightedAverageControls(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class RandomWeightedAverageImages(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

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
        self.gradient_penalty_weight = 10

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

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty) 

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
        pass
    
    def train(self, batch_size=50, epochs=100, weights_name="default_weights"):

        self.num_control_outputs = len(self.info["key_labels"])

        optimizer_g = optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
        optimizer_d = optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
        optimizer_conv = optimizers.Adam()


        
        generator = self.create_generator()
        discriminator = self.create_discriminator()


        print("########### GENERATOR ############")
        for layer in discriminator.layers:
            layer.trainable = False
        discriminator.trainable = False
        noise = Input(shape=(self.noise_dim,))
        img = Input(shape=(self.img_h, self.img_w, self.img_d))
        generator_layers = generator([img, noise])
        discriminator_layers_for_generator = discriminator([img, generator_layers])
        self.generator = Model(inputs=[img, noise], outputs=[discriminator_layers_for_generator])
        self.generator.compile(loss=self.wasserstein_loss, 
                               optimizer=optimizer_g)
        self.generator.summary()

        for layer in discriminator.layers:
            layer.trainable = True
        for layer in generator.layers:
            layer.trainable = False
        discriminator.trainable = True
        generator.trainable = False

        

        print("\n\n########### DISCRIMINATOR ############")
        generator_img_input_for_discriminator = Input(shape=(self.img_h, self.img_w, self.img_d), name="generator_img_input_for_discriminator")
        generator_noise_input_for_discriminator = Input(shape=(self.noise_dim,), name="generator_noise_input_for_discriminator")
        generated_controls_for_discriminator = generator([generator_img_input_for_discriminator, generator_noise_input_for_discriminator])
        discriminator_output_from_generator = discriminator([generator_img_input_for_discriminator, generated_controls_for_discriminator])

        real_controls = Input(shape=(self.num_control_outputs,), name="real_controls")
        real_disc_img_input = Input(shape=(self.img_h, self.img_w, self.img_d), name="real_disc_img_input")
        discriminator_output_from_real_samples = discriminator([real_disc_img_input, real_controls])
        
        averaged_controls = RandomWeightedAverageControls(batch_size)([real_controls, generated_controls_for_discriminator])
        averaged_imgs = RandomWeightedAverageImages(batch_size)([real_disc_img_input, generator_img_input_for_discriminator])
        averaged_controls_out = discriminator([averaged_imgs, averaged_controls])
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=averaged_controls,
                                  gradient_penalty_weight=self.gradient_penalty_weight)
        partial_gp_loss.__name__ = "gradient_penalty"
        
        self.discriminator = Model(inputs=[real_disc_img_input, real_controls,
                                           generator_img_input_for_discriminator, generator_noise_input_for_discriminator],
                                   outputs=[discriminator_output_from_real_samples,
                                            discriminator_output_from_generator,
                                            averaged_controls_out])
        self.discriminator.compile(optimizer=optimizer_d,
                                   loss=[self.wasserstein_loss,
                                         self.wasserstein_loss,
                                         partial_gp_loss])

        self.generator.summary()
        self.discriminator.summary()



        X, y = shuffle(self.X, self.y, random_state=0)
        
        split_idx = int(X.shape[0] * 0.8)

        X_train, X_val = X[:split_idx,:,:,:], X[split_idx:,:,:,:]
        y_train, y_val = y[:split_idx,:], y[split_idx:,:]

        weights_path = self.model_path / "weights" / weights_name
        try_make_dirs(weights_path)

        losses = []

        positive_y = np.ones((batch_size, 1), dtype="float32")
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype="float32")
        
        for epoch in range(epochs):
            X_train, y_train = shuffle(X_train, y_train, random_state=0)

            d_losses = []
            g_losses = []
            minibatches_size = batch_size * self.n_critic
            iterations = int(X_train.shape[0] // (batch_size * self.n_critic))
    
            for n_iter in range(iterations):
                X_disc_minibatch = X_train[n_iter * minibatches_size:(n_iter+1) * minibatches_size]
                y_disc_minibatch = y_train[n_iter * minibatches_size:(n_iter+1) * minibatches_size]
                for n_critic_iter in range(self.n_critic):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    real_controls = y_disc_minibatch[n_critic_iter * batch_size:(n_critic_iter+1) * batch_size]
                    imgs = X_disc_minibatch[n_critic_iter * batch_size:(n_critic_iter+1) * batch_size]
                    noise = np.random.rand(batch_size, self.noise_dim).astype(np.float32)
                    d_loss = self.discriminator.train_on_batch([imgs, real_controls, imgs, noise],
                                                                [positive_y, negative_y, dummy_y])
                    d_losses.append(d_loss)
                # ---------------------
                #  Train Generator
                # ---------------------
                noise = np.random.rand(batch_size, self.noise_dim).astype(np.float32)
                imgs = X_train[n_iter * batch_size: (n_iter + 1) * batch_size]
                g_loss = self.generator.train_on_batch([imgs, noise],
                                                        positive_y)
                g_losses.append(g_loss)
                print("Iteration {}/{} - D_loss_1: {:.6f} - D_loss_2: {:.6f} - D_loss_3: {:.6f} - D_loss_4: {:.6f} -  G_loss: {:.6f}".format(n_iter, iterations, *d_loss, g_loss))            

#            d_val_losses = []
#            g_val_losses = []
#            val_iterations = math.ceil(X_val.shape[0] / batch_size)
#            for n_val_iter in range(val_iterations):
#                val_batch_start_idx = n_val_iter * batch_size
#                if (n_val_iter + 1) * batch_size >= X_val.shape[0]:
#                    val_batch_end_idx = X_val.shape[0]
#                else:
#                    val_batch_end_idx = (n_val_iter + 1) * batch_size
#
#                X_val_batch = X_val[val_batch_start_idx:val_batch_end_idx,:,:,:]
#                y_val_batch = y_val[val_batch_start_idx:val_batch_end_idx,:]
#
#                noise = np.random.normal(0, 1, (X_val_batch.shape[0], self.noise_dim))
#                valid = np.ones((X_val_batch.shape[0], 1), dtype="float32")
#                fake = -valid
#
#                gen_controls = self.generator.predict([X_val_batch, noise])
#
#                d_loss_real_val = self.discriminator.test_on_batch([X_val_batch, y_val_batch], fake)
#                d_loss_fake_val = self.discriminator.test_on_batch([X_val_batch, gen_controls], valid)
#                d_loss_val = 0.5 * np.add(d_loss_fake_val, d_loss_real_val)
#                g_loss_val = self.combined.test_on_batch([X_val_batch, noise], fake)
#
#                d_val_losses.append(d_loss_val)
#                g_val_losses.append(g_loss_val)
#
#            d_loss_val = np.mean(d_val_losses)
#            g_loss_val = np.mean(g_val_losses)
#            avg_g_loss = np.mean(g_losses)
#            avg_d_loss = np.mean(d_losses)
#
#            print("Epoch {}/{} AVG_D_loss: {:.6f} - AVG_G_loss: {:.6f} - D_loss_val: {:.6f} - G_loss_val: {:.6f}".format(epoch + 1, epochs, avg_d_loss, avg_g_loss, d_loss_val, g_loss_val))
#
#            losses.append([avg_d_loss, avg_g_loss, d_loss_val, g_loss_val])
#            with open(weights_path / "loss.csv", "w") as f:
#                writer = csv.writer(f)
#                writer.writerows(losses)
            epoch_suffix = "epoch-{}".format(epoch + 1)
            try_make_dirs(weights_path / epoch_suffix)

            self.generator.save_weights(weights_path / epoch_suffix / "generator_weights.h5")
            self.discriminator.save_weights(weights_path / epoch_suffix / "discriminator_weights.h5")
            with open(weights_path / epoch_suffix / "info.json", "w") as info_file:
                json.dump(self.info, info_file)
            

        self.generator.save_weights(weights_path / "generator_weights.h5")
        self.discriminator.save_weights(weights_path / "discriminator_weights.h5")
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
        



