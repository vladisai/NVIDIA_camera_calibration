import numpy as np
import keras
import dataset_loader

class SingleDsDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size, columns):
        'Initialization'
        self.path = path 
        self.files_count = 0
        self.batch_size = batch_size
        self.seek = 0
        self.read_length = 0
        self.columns = columns
        self.in_file_index = 0
        self.__preprocess()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.total_length) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        if self.current_X = None:
            self.__read_files(self.file_index)

        X, Y = self.__get_rows(self.batch_size)
        while len(Y) < self.batch_size:
            if self.in_file_index == len(self.current_X):
                self.file_index += 1
                self.file_index %= self.files_count
                self.__read_files(self.file_index)
            X_c, Y_c = self.__get_rows(self.batch_size - len(X))
            X = np.concatenate(X, X_c)
            Y = np.concatenate(Y, Y_c)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.file_index = 0
        self.current_X = None
        self.current_Y = None

    def __get_xy_names(self, i):
        return os.path.join(self.path, 'X_{}.npy'.format(i)), os.path.join(self.path, 'Y_{}.csv'.format(i))

    def __preprocess(self):
        files = os.listdir(self.path)
        self.files_count = len(files) / 2
        self.total_length = 0
        for i in range(self.files_count):
            _, y = self.__get_xy_names(i)
            Y = pd.read_csv(y)
            self.total_length += len(Y)

    def __read_files(self, index):
        # TODO: check if this works
        self.current_X, self.current_Y = dataset_loader.loadXY('', self.path, 1, self.columns, index)
        self.in_file_index = 0

    def __get_rows(self, k):
        # gets at most k rows from the file
        X = self.current_X[self.in_file_index:min(len(self.current_X), self.in_file_index + k)]
        Y = self.current_Y[self.in_file_index:min(len(self.current_Y), self.in_file_index + k)]
        self.in_file_index += len(Y)
        return X, Y


