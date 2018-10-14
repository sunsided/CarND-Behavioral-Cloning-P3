# coding: utf-8
# This model.py file was created solely to fulfill all requirements for the CarND Behavioral Cloning project;
# the code of this script is however taken almost exactly from the training.ipynb Jupyter Notebook.
# I encourage you to look at it instead for explanations and insights.

import os
import random
from typing import Tuple

import PIL.Image as Image
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Dense, Lambda, Cropping2D, SeparableConv2D, Dropout, PReLU
from keras.models import Sequential
from keras.optimizers import Adam
from training_data import DATASET_PATH, load_training_data, DataGenerator

MODEL_PATH = 'models'

# Load the training, validation and test data
df_train, df_test, df_valid = load_training_data()

# In order to add some noise to the steering angles, we'll be using the "natural" standard deviation
# of the data.
steering_std = df_train['Steering Angle'].std()

# Since the sky region is not containing any information relevant to steering, we can remove it.
# A possible exception to this assumption would be situations where the car is driving down a hill;
# it might be worth exploring that later.
# Likewise, the bottom part of the image always contains the hood of the car and is thus not relevant to steering.
CROP_TOP_BIG = 75
CROP_BOTTOM_BIG = 22

# The core changes we'll be doing to the NVIDIA network are the following:
#
# - The original image size is used, i.e. resizing is removed,
# - 2D convolutions are replaced with depthwise separable 2D convolutions,
# - ReLU activations are replaced with Parametric ReLU, and
# - Dropout is added before the first fully connected layer.
#
# The removal of the resizing step is done purely to improve processing speed; at the same time, the introduction of
# depthwise separable convolutions should allow for some reduction of parameters as well. Much more importantly, using
# a depthwise separable convolution in the input layer follows the intuition, that grayscale intensity matters much
# more to lane detection that color. By learning filters individually for the luminance (Y) and chrominance (U and V)
# channels, it should be possible to focus the training on the more important channels rather than trying to use them
# simultaneously. Separable convolutions are kept for later layers because of the same reason; a rather efficient class
# of neural networks designed for mobile use (such as embedded systems) using separable convolutions are
# MobileNets (https://arxiv.org/abs/1704.04861).
#
# ReLU activations drop all negative activations, effectively cancelling out their gradient, possibly leading to "dead"
# neurons during training (when a neuron's gradient is zero, no learning can happen ever again). While this allows for
# pruning of the network, it possibly throws aways computational capabilities of the architecture. Hence,
# Parametric ReLU units are used instead: Functioning like Leaky ReLU, a class of activation functions that "leaks"
# negative activations by using a small coefficient on them, Parametric ReLUs allow for learning the influence of
# negative activations as well at the cost of one extra trainable parameter per activation function. Nowadays,
# activation functions such as ELU and SELU have proven to be more efficient in terms of training and inference quality;
# they do rely on exponential functions though, and since the goal is to have a network that should run as fast as
# possible on an embedded device, I decided not to use them.
#
# Lastly, dropout is applied exactly once. Since dropout randomly "drops" connections during training, the network has
# to learn redundancy. When repeatedly using dropouts in subsequent layers, this only becomes worse. Again, pruning may
# mitigate this effect by simply removing "identical" neurons, but this is beyond the scope of this project, more than
# one instance of dropout in the network was deemed enough.
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


def custom_model(learning_rate: float, decay: float = 0, dropout_rate: float = 0.3):
    model = Sequential()

    # YUV conversion and normalization
    model.add(Lambda(yuv_conversion, input_shape=(160, 320, 3), name='rgb_to_yuv'))
    model.add(Lambda(lambda x: x * 2 - 1, name='normalize'))

    # Crop the image to remove the sky and car hood
    model.add(Cropping2D(cropping=((CROP_TOP_BIG, CROP_BOTTOM_BIG), (0, 0)),
                         name='crop'))

    model.add(SeparableConv2D(24, (5, 5), strides=(3, 3), padding='valid',
                              depth_multiplier=16, name='conv_1'))
    model.add(PReLU(name='conv_1_prelu'))

    model.add(SeparableConv2D(36, (5, 5), strides=(3, 3), padding='valid',
                              depth_multiplier=9, name='conv_2'))
    model.add(PReLU(name='conv_2_prelu'))

    model.add(SeparableConv2D(48, (3, 3), strides=(2, 2), padding='valid',
                              depth_multiplier=6, name='conv_3'))
    model.add(PReLU(name='conv_3_prelu'))

    model.add(SeparableConv2D(64, (2, 2), strides=(2, 2), padding='valid',
                              depth_multiplier=9, name='conv_4'))
    model.add(PReLU(name='conv_4_prelu'))

    model.add(Flatten())
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(100, name='fc_1'))
    model.add(PReLU(name='fc_1_prelu'))

    model.add(Dense(50, name='fc_2'))
    model.add(PReLU(name='fc_2_prelu'))

    model.add(Dense(10, name='fc_3'))
    model.add(PReLU(name='fc_3_prelu'))

    model.add(Dense(1, name='angle'))

    adam = Adam(lr=learning_rate, decay=decay)
    model.compile(optimizer=adam, loss='mse')

    return model


model = custom_model(LEARNING_RATE)
print(model.summary())


class AugmentingDataGenerator2(DataGenerator):
    'Generates data for Keras'

    def __init__(self, df: pd.DataFrame, batch_size: int = 32, dim: Tuple[int, int] = (320, 160, 3), shuffle=True,
                 dataset_path: str = DATASET_PATH, steering_std: float = 0, shift_correction: float = 0.2):
        # Combine the training examples
        df_c = df[['Center Image', 'Steering Angle']].rename(columns={'Center Image': 'Image'})
        df_l = df[['Left Image', 'Steering Angle']].rename(columns={'Left Image': 'Image'})
        df_r = df[['Right Image', 'Steering Angle']].rename(columns={'Right Image': 'Image'})
        df_l['Steering Angle'] += shift_correction
        df_r['Steering Angle'] -= shift_correction
        df = pd.concat([df_c, df_l, df_r], axis=0, sort=False)

        super().__init__(df, batch_size, dim, shuffle, dataset_path)
        self.steering_std = steering_std
        self.shift_correction = shift_correction

    def _get_example(self, row) -> Tuple[np.ndarray, np.float32]:
        path = os.path.join(self.dataset_path, row['Image'])
        angle = row['Steering Angle']
        angle += np.random.normal(scale=self.steering_std)

        img = Image.open(path).convert('RGB')
        if random.choice([True, False]):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angle = -angle
        X = np.array(img).astype(np.uint8)
        y = angle
        return X, y


# We can now create the training and validation data generators.
training_generator = AugmentingDataGenerator2(df_train, batch_size=BATCH_SIZE,
                                              steering_std=steering_std / 2,
                                              shift_correction=0.2)
validation_generator = DataGenerator(df_valid, batch_size=BATCH_SIZE)

# We'll be using a `ModelCheckpoint` to save a model whenever the validation loss decreases.
# An issue exists for Keras 2.2.4 that prevents writing checkpoint files when `ModelCheckpoint` is used with
# `fit_generator()` and `use_multiprocessing=True` (see [here](https://github.com/keras-team/keras/issues/11101)).
# One suggested workaround is to use formatted file names, which is what we'll do.
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'custom-1.{epoch:02d}-{val_loss:.4f}.h5')
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
