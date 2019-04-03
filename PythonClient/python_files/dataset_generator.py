import numpy as np
import keras
import dataset_loader
import os
import pandas as pd
import random

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, apply_affine_transform

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size, columns, seek=0, read_length=10, augment=False):
        self.path = path 
        self.files_count = 0
        self.batch_size = batch_size
        self.seek = seek 
        self.read_length = read_length
        self.columns = columns
        self.in_file_index = 0
        self.augment = augment
        self.__preprocess()
        self.on_epoch_end()
        if augment:
            self.AUGMENTATIONS_PER_IMAGE = 1
        else:
            self.AUGMENTATIONS_PER_IMAGE = 0

    def __len__(self):
        return int(np.floor(self.total_length * (1 + self.AUGMENTATIONS_PER_IMAGE) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        if self.current_X is None or self.in_file_index == len(self.current_X):
            self.__read_files(self.file_index)

        X, Y = self.__get_rows(self.batch_size)
        while len(Y) < self.batch_size:
            if self.in_file_index == len(self.current_X):
                self.file_index += 1
                if self.file_index == self.seek + self.read_length:
                    self.on_epoch_end()
                    break
                self.__read_files(self.file_index)
            X_c, Y_c = self.__get_rows(self.batch_size - len(X))
            X = np.append(X, X_c, 0)
            Y = Y.append(Y_c)

        def foo(x):
            if x > 0.3:
                return 1
            if x < 0.3:
                return -1
            return 0

        sign = Y.loc[:, "steer"].values
        sign = np.array([foo(xi) for xi in sign]).astype(np.float64)
        sign = sign[:, np.newaxis]

        speed = (Y.loc[:, "speed"].values).astype(np.float64)
        speed = speed[:, np.newaxis]
        meta = speed
        X = [X, meta]
        Y = Y[self.columns].values
        return X, Y

    def on_epoch_end(self):
        self.file_index = self.seek 
        self.current_X = None
        self.current_Y = None
        print('on epoch end')

    def __get_xy_names(self, i):
        return os.path.join(self.path, 'X_{}.npy'.format(i)), os.path.join(self.path, 'Y_{}.csv'.format(i))

    def __preprocess(self):
        files = os.listdir(self.path)
        self.files_count = len(files) // 4
        self.total_length = 0

        for i in range(self.seek, self.seek + self.read_length):
            _, y = self.__get_xy_names(i)
            Y = pd.read_csv(y)
            self.total_length += len(Y)
        print('total length is ', self.total_length)

    def __read_files(self, index):
        self.current_X, self.current_Y = dataset_loader.loadXY('', self.path, 1, suffix=index)
        if self.augment:
            X, Y = dataset_loader.loadXY('', self.path, 1, suffix='{}_big'.format(index))
            X_aug, Y_aug = [], []
            for img, measurement in zip(X, Y.iterrows()):
                measurement = measurement[1] # cut tuple index out
                for _ in range(self.AUGMENTATIONS_PER_IMAGE):
                    x, y = random_augmentation(img, measurement)
                    X_aug.append(x)
                    Y_aug.append(y)
            X_aug = np.stack(X_aug)
            Y_aug = pd.DataFrame(Y_aug)
            self.current_X = np.concatenate([self.current_X, X_aug])
            self.current_Y = pd.concat([self.current_Y, Y_aug], ignore_index=True)
            perm = np.random.permutation(len(self.current_X))
            self.current_X = self.current_X[perm]
            self.current_Y = pd.DataFrame(data=self.current_Y.values[perm], columns=self.current_Y.columns)
            
        self.in_file_index = 0

    def __get_rows(self, k):
        # gets at most k rows from the file
        X = self.current_X[self.in_file_index:min(len(self.current_X), self.in_file_index + k)]
        Y = self.current_Y[self.in_file_index:min(len(self.current_Y), self.in_file_index + k)]
        self.in_file_index += len(Y)
        return X, Y

def crop(x, width=160, height=120):
    height_x = x.shape[0]
    width_x = x.shape[1]
    margin_x_left = (width_x - width) // 2
    margin_x_right = (width_x - width + 1) // 2
    margin_y_top = (height_x - height) // 2
    margin_y_bottom = (height_x - height + 1) // 2
    xc = x[margin_y_top:-margin_y_bottom:, margin_x_left:-margin_x_right]
    return xc

def shear(xa, t):
    xb = apply_affine_transform(xa, theta=-t, shear=t, fill_mode='nearest')
    xb = apply_affine_transform(xb, ty=t*1.7, fill_mode='nearest')
    return xb

def shift(xa, t):
    xb = apply_affine_transform(xa, ty=t, fill_mode='nearest')
    return xb

def random_augmentation(x, y):
    steer_diff = 0
    if random.random() < 0.5:
        shift_x = random.randint(-30, 30)
        steer_diff = shift_x / 40 * 0.3
        x = shift(x, shift_x)
    else:
        shear_x = random.randint(-30, 30)
        steer_diff = shear_x / 40 * 0.5
        x = shear(x, shear_x)
    x = crop(x)
    y['steer'] += steer_diff
    return x, y
