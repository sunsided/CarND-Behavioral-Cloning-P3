# coding: utf-8

import os
from typing import Tuple

import PIL.Image as Image
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


DATASET_PATH = os.path.join('.', 'dataset')

DRIVING_LOG_FILENAME = 'driving_log.csv'
DRIVING_LOG_PATH = os.path.join(DATASET_PATH, DRIVING_LOG_FILENAME)


def load_training_data(driving_log_path: str=DRIVING_LOG_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(driving_log_path, header=None,
                     names=['Center Image', 'Left Image', 'Right Image',
                            'Steering Angle',
                            'Throttle', 'Brake', 'Speed'])

    df['Steering Angle'] = df['Steering Angle'].astype(np.float32)
    df['Throttle']       = df['Throttle'].astype(np.float32)
    df['Brake']          = df['Brake'].astype(np.float32)
    df['Speed']          = df['Speed'].astype(np.float32)

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

    return df_train, df_valid, df_test


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


