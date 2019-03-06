import numpy as np
import keras
import dataset_loader
import os
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size, columns, seek=0, read_length=10):
        self.path = path 
        self.files_count = 0
        self.batch_size = batch_size
        self.seek = seek 
        self.read_length = read_length
        self.columns = columns
        self.in_file_index = 0
        self.__preprocess()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.total_length / self.batch_size))

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
        print('finished')
        return X, Y

    def on_epoch_end(self):
        self.file_index = self.seek 
        self.current_X = None
        self.current_Y = None

    def __get_xy_names(self, i):
        return os.path.join(self.path, 'X_{}.npy'.format(i)), os.path.join(self.path, 'Y_{}.csv'.format(i))

    def __preprocess(self):
        files = os.listdir(self.path)
        self.files_count = len(files) // 2
        self.total_length = 0

        for i in range(self.seek, self.seek + self.read_length):
            _, y = self.__get_xy_names(i)
            Y = pd.read_csv(y)
            self.total_length += len(Y)
        print('total length is ', self.total_length)

    def __read_files(self, index):
        self.current_X, self.current_Y = dataset_loader.loadXY('', self.path, 1, index=index)
        self.in_file_index = 0

    def __get_rows(self, k):
        # gets at most k rows from the file
        X = self.current_X[self.in_file_index:min(len(self.current_X), self.in_file_index + k)]
        Y = self.current_Y[self.in_file_index:min(len(self.current_Y), self.in_file_index + k)]
        self.in_file_index += len(Y)
        return X, Y


