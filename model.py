# coding: utf-8
# This model.py file was created solely to fulfill all requirements for the CarND Behavioral Cloning project;
# the code of this script is however taken almost exactly from the training.ipynb Jupyter Notebook.
# I encourage you to look at it instead for explanations and insights.

import os
import random
from typing import Tuple

import PIL.Image as Image
import keras
import numpy as np
import pandas as pd
from PIL.Image import Image
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Dense, Lambda, Activation, Conv2D, Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MODEL_PATH = 'models'
DATASET_PATH = os.path.join('.', 'dataset')

DRIVING_LOG_FILENAME = 'driving_log.csv'
DRIVING_LOG_PATH = os.path.join(DATASET_PATH, DRIVING_LOG_FILENAME)

df = pd.read_csv(DRIVING_LOG_PATH, header=None, 
                 names=['Center Image', 'Left Image', 'Right Image', 
                        'Steering Angle', 
                        'Throttle', 'Brake', 'Speed'])

df['Steering Angle'] = df['Steering Angle'].astype(np.float32)
df['Throttle']       = df['Throttle'].astype(np.float32)
df['Brake']          = df['Brake'].astype(np.float32)
df['Speed']          = df['Speed'].astype(np.float32)

steering_std = df['Steering Angle'].std()

# We're going to sample data based on zero or nonzero steering angle.
# We're also generally only interested in rows with a positive speed.
zero_angles = (df['Steering Angle'] == 0)
positive_speed = (df['Speed'] > 0)

# We now split the dataset into two groups: Zero and nonzero steering angles.
sample_seed = 0
df_nonzero_angles = df[positive_speed & ~zero_angles]
df_zero_angles    = df[positive_speed & zero_angles]

# We're going to split both sets into two groups once more: Training and validation data.
df_train_nonzero, df_valid_nonzero = train_test_split(df_nonzero_angles, 
                                                      test_size=0.3, random_state=sample_seed)
df_train_zero, df_valid_zero       = train_test_split(df_zero_angles, 
                                                      test_size=0.3, random_state=sample_seed)

# We now split the validation set again into the actual validation set (used during training)
# and a test set (used to validate the final model).
df_test_nonzero, df_valid_nonzero = train_test_split(df_valid_nonzero, 
                                                     test_size=0.3, random_state=sample_seed)
df_test_zero, df_valid_zero       = train_test_split(df_valid_zero, 
                                                     test_size=0.3, random_state=sample_seed)

# The "zero" training dataset still has too many elements. We're going to take a random sample of it for training.
zero_speed_factor = 0.1
zero_speed_count_train = int(zero_speed_factor * len(df_train_zero))
zero_speed_count_valid = int(zero_speed_factor * len(df_valid_zero))
zero_speed_count_test  = int(zero_speed_factor * len(df_test_zero))

df_train_zero = df_train_zero.sample(zero_speed_count_train, random_state=sample_seed)
df_valid_zero = df_valid_zero.sample(zero_speed_count_valid, random_state=sample_seed)
df_test_zero  = df_test_zero.sample(zero_speed_count_test, random_state=sample_seed)

# We now combine the data into the actual data sets used for training.
df_train = pd.concat([df_train_zero, df_train_nonzero])
df_valid = pd.concat([df_valid_zero, df_valid_nonzero])
df_test = pd.concat([df_test_zero, df_test_nonzero])


# For the first iteration of training, we're only going to train with images from the center camera.
# We'll be using a custom data generator as described in A detailed example of how to use data generators with Keras
# (https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly).
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, df: pd.DataFrame, batch_size: int=32, dim: Tuple[int, int]=(320, 160, 3), shuffle=True,
                dataset_path: str=DATASET_PATH):
        self.batch_size = batch_size
        self.df = df
        self.dim = dim
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        rows = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(rows)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            # self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.df = shuffle(self.df)

    def __data_generation(self, rows) -> Tuple[np.ndarray, np.float32]:
        """Generates data containing batch_size samples"""
        # Initialization
        width, height, channels = self.dim
        X = np.empty((self.batch_size, height, width, channels), dtype=np.uint8)
        y = np.empty((self.batch_size), dtype=np.float32)
        
        # Generate data
        for i, (idx, row) in enumerate(rows.iterrows()):
            # Store sample and angle
            X[i,], y[i] = self._get_example(row)

        return X, y
    
    def _get_example(self, row) -> Tuple[np.ndarray, np.float32]:
        path = os.path.join(self.dataset_path, row['Center Image'])
        img = Image.open(path).convert('RGB')
        X = np.array(img).astype(np.uint8)
        y = row['Steering Angle']
        return X, y


# Since the sky region is not containing any information relevant to steering, we can remove it.
# A possible exception to this assumption would be situations where the car is driving down a hill;
# it might be worth exploring that later.
# Likewise, the bottom part of the image always contains the hood of the car and is thus not relevant to steering.
# If we decide to use images from the left and right camera as well, this region might be problematic as well,
# as the network could learn to be biased towards the position of the hood.
NVIDIA_WIDTH  = 200
NVIDIA_HEIGHT = 66
NVIDIA_ASPECT = NVIDIA_HEIGHT / NVIDIA_WIDTH

NEW_WIDTH = int(NVIDIA_WIDTH)
NEW_HEIGHT = int(320 * NVIDIA_ASPECT)

CROP_TOP = 24
CROP_BOTTOM = 15

# As a baseline, we're going to implement the model suggested by the End to End Learning for Self-Driving Cars paper
# (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
# published by NVIDIA.
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3


# The paper specifies image input is converted to YUV prior to processing.
def yuv_conversion(x):
    import tensorflow as tf
    x = tf.cast(x, dtype=tf.float32) / 255.0
    return tf.image.rgb_to_yuv(x)


# In order to be as close to the original paper as possible, we'll be resizing the image to 200 pixels in width
# such that after cropping, the input image size will `200x66`.
def image_resize(x):
    return K.tf.image.resize_images(x, (105, 200))


def nvidia_model(learning_rate: float, decay: float=0):
    model = Sequential()

    # Resizing the images to the NVIDIA proposed shape:
    model.add(Lambda(image_resize, input_shape = (160, 320, 3), name='resize_image'))
    
    # YUV conversion and normalization
    model.add(Lambda(yuv_conversion, input_shape = (105, 200, 3), name='rgb_to_yuv'))
    model.add(Lambda(lambda x: x * 2 - 1, input_shape = (105, 200, 3), name='normalize'))
    
    # Crop the image to remove the sky and car hood
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0)), 
                         input_shape=(160, 320, 3), name='crop'))
    
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', name='conv_1'))
    model.add(Activation('relu', name='conv_1_relu'))
    
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', name='conv_2'))
    model.add(Activation('relu', name='conv_2_relu'))
    
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', name='conv_3'))
    model.add(Activation('relu', name='conv_3_relu'))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', name='conv_4'))
    model.add(Activation('relu', name='conv_4_relu'))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', name='conv_5'))
    model.add(Activation('relu', name='conv_5_relu'))

    # In this setup, the number of parameters after flattening (`1152`) differs from the NVIDIA paper (`1164`).
    # However, since the paper explicitly states a convolution output of `64@1x18` was obtained,
    # the number reported by NVIDIA appears to be an error.
    model.add(Flatten())    

    model.add(Dense(100, name='fc_1'))
    model.add(Activation('relu', name='fc_1_relu'))
    
    model.add(Dense(50, name='fc_2'))
    model.add(Activation('relu', name='fc_2_relu'))
    
    model.add(Dense(10, name='fc_3'))
    model.add(Activation('relu', name='fc_3_relu'))

    model.add(Dense(1, name='angle'))

    adam = Adam(lr=learning_rate, decay=decay)
    model.compile(optimizer=adam, loss='mse') 
    
    return model


model = nvidia_model(LEARNING_RATE)
print(model.summary())

# We can now create the training and validation data generators.
training_generator   = DataGenerator(df_train, batch_size=BATCH_SIZE)
validation_generator = DataGenerator(df_valid, batch_size=BATCH_SIZE)

# We'll be using a `ModelCheckpoint` to save a model whenever the validation loss decreases.
# An issue exists for Keras 2.2.4 that prevents writing checkpoint files when `ModelCheckpoint` is used with
# `fit_generator()` and `use_multiprocessing=True` (see [here](https://github.com/keras-team/keras/issues/11101)).
# One suggested workaround is to use formatted file names, which is what we'll do.
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'nvidia.{epoch:02d}-{val_loss:.4f}.h5')
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor = 'val_loss', verbose=1,
                             save_best_only=True, mode='min')

# In addition, early stopping will be used to terminate training if the validation loss
# doesn't improve for multiple epochs.
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                               mode='min', restore_best_weights=False)

# We can now run the training.
hist = model.fit_generator(generator=training_generator,
                           validation_data=validation_generator,
                           use_multiprocessing=True, workers=6,
                           callbacks=[checkpoint, early_stopping],
                           epochs=EPOCHS, verbose=1)


# When looking at the training and validation losses, we find that both appeared to still decrease at the time training
# was early stopped. Training loss is still a fair amount from zero and validation loss didn't go up, so overfitting
# doesn't seem to be an issue yet. We might want to train for more epochs (e.g. using more patience for early stopping)
# and possibly use a lower learning rate.

# Some of the inferred steering angles in the video above are a bit problematic; ignoring the fact that my driving
# instructions on the simulator weren't too smooth either, we can try adding more data and a somewhat noisy steering
# angle to force the network to generalize more.
# A simple augmentation we can do is randomly flipping the images and inverting the steering angle. Another trick
# proposed by the NVIDIA paper is to make use of two additional cameras installed to the left and right of the car's
# center. This basically simulates slightly offset car positions, introducing more training data. If we assume a
# destination angle as determined by the center camera, due to trigonometry, we need to add correction factor to the
# angle whenever we're using either the left or right image, positive and negative respectively. Since no reference
# values for distances are given, we'll be simply using an arbitrarily selected value of e.g. `0.2`.

# To prevent overfitting, a zero-mean gaussian noise is added to the recorded steering angles
# during training additionally.
class AugmentingDataGenerator(DataGenerator):
    'Generates data for Keras'
    def __init__(self, df: pd.DataFrame, batch_size: int=32, dim: Tuple[int, int]=(320, 160, 3), shuffle=True,
                dataset_path: str=DATASET_PATH, steering_std: float=0, shift_correction: float=0.2):
        super().__init__(df, batch_size, dim, shuffle, dataset_path)
        self.steering_std = steering_std
        self.shift_correction = shift_correction
    
    def _get_example(self, row) -> Tuple[np.ndarray, np.float32]:
        i = random.randint(0, 2)
        r = random.randint(0, 1)
        if i == 0:
            path = os.path.join(self.dataset_path, row['Center Image'])
            correction = 0
        elif i == 1:
            path = os.path.join(self.dataset_path, row['Left Image'])
            correction = +self.shift_correction
        else:
            path = os.path.join(self.dataset_path, row['Right Image'])
            correction = -self.shift_correction
        
        angle = row['Steering Angle'] + correction
        angle += np.random.normal(scale=self.steering_std)
        
        img   = Image.open(path).convert('RGB')
        if r == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angle = -angle
        X = np.array(img).astype(np.uint8)
        y = angle
        return X, y


# With this, we restart the training.
K.clear_session()
model = nvidia_model(LEARNING_RATE)

training_generator   = AugmentingDataGenerator(df_train, batch_size=BATCH_SIZE, 
                                               steering_std=steering_std,
                                               shift_correction=0.2)
validation_generator = DataGenerator(df_valid, batch_size=BATCH_SIZE)

CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'nvidia-aug.{epoch:02d}-{val_loss:.4f}.h5')
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor = 'val_loss', verbose=1,
                             save_best_only=True, mode='min')

early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                               mode='min', restore_best_weights=False)

hist = model.fit_generator(generator=training_generator,
                           validation_data=validation_generator,
                           use_multiprocessing=True, workers=6,
                           callbacks=[checkpoint, early_stopping],
                           epochs=EPOCHS, verbose=1)
