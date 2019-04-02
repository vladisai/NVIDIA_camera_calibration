try:
    from python_files import dataset_loader
except:
    import dataset_loader

import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_absolute_error

import time

class ModelBase(ABC):
    def __init__(self, input_shape, output_length = 1):
        self.input_shape = input_shape
        self.output_length = output_length
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self.model.fit_generator(*args, **kwargs)

    def evaluate_generator(self, *args, **kwargs):
        return self.model.evaluate_generator(*args, **kwargs)

    def train(self, X, Y, X_val = None, Y_val = None, epochs = 1, batch_size = 64):
        t = time.time()
        for j in range(0, epochs):
            print('epoch {}'.format(j))
            tt = time.time()
            mae = 0
            for i in range(0, len(X), batch_size):
                X_batch = dataset_loader.transform_batch(X[i:(i + batch_size)], self.input_shape)
                Y_batch = Y[i:(i + batch_size)]
                if i + batch_size < len(X):
                    mae += self.model.train_on_batch(X_batch, Y_batch)[0]
            print('mae is {}'.format(mae / (len(X) / batch_size)))
            print('last epoch took {} s'.format(time.time() - tt))
            print('eta is  {} s'.format((time.time() - t) / (j + 1) * (epochs - j - 1)))
            if X_val is not None:
                print('validation mae is {}'.format(self.validate(X_val, Y_val)))

    def predict_raw(self, X, batch_size = 10, verbose=False):
        Y = np.zeros((len(X), 1))
        t = time.time()
        for i in range(0, len(X), batch_size):
            X_batch = dataset_loader.transform_batch(X[i:(i + batch_size)], self.input_shape)
            Y[i:min(i + batch_size, len(Y))] = self.predict(X_batch)
            if verbose: 
                print('{} images, eta is  {} s'.format(i, (time.time() - t) / (i + 1) * (len(X) - i - 1)))
        return Y

    def predict(self, X):
        return self.model.predict(X)

    def validate(self, X, Y, verbose=False):
        pred = self.predict(X)
        for i in range(0, self.output_length):
            if verbose:
                print('{}th feature is {}'.format(i, mean_absolute_error(pred[:, i], Y[:, i])))
        return mean_absolute_error(pred, Y)

    def save(self, filename, save_weights=True):
        model_json = self.model.to_json()
        with open("{}.json".format(filename), "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
                if save_weights:
                    self.model.save_weights("{}.h5".format(filename))
                print("Saved model to disk")

    def load(self, filename):
        self.model.load_weights("{}.h5".format(filename))

class ModelSimple3(ModelBase):
    def build_model(self):
        model = Sequential()

        convnet = Sequential()
        convnet.add(Conv2D(24, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape,
                         data_format="channels_last"))
        convnet.add(Conv2D(48, (3, 3), activation='relu'))
        convnet.add(MaxPooling2D(pool_size=(2, 2)))
        convnet.add(Conv2D(64, (5, 5), activation='relu'))
        convnet.add(MaxPooling2D(pool_size=(2, 2)))
        convnet.add(Dropout(0.25))
        convnet.add(Flatten())

        extra = Sequential()
        extra.add(Activation('linear', input_shape=(1)))

        model.add(Merge([convnet, extra], mode='concat', concat_axis=2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.output_length, activation='linear'))

        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        return model

class ModelSimple2(ModelBase):
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(24, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape,
                         data_format="channels_last"))
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.output_length, activation='linear'))

        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        return model

class ModelSimpleMSE(ModelBase):
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(24, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape,
                         data_format="channels_last"))
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.output_length, activation='linear'))

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model

class ModelSimple(ModelBase):
    def build_model(self):

        img = Input(shape=self.input_shape)
        meta = Input(shape=(1,))

        l = Conv2D(24, kernel_size=(3, 3),
                         activation='relu',
                         data_format="channels_last")(img)
        l = BatchNormalization()(l)
        l = Conv2D(48, (3, 3), activation='relu')(l)
        l = BatchNormalization()(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = Flatten()(l)

        l = keras.layers.concatenate([l, meta], axis=-1)

        l = Dense(128, activation='relu')(l)
        l = BatchNormalization()(l)
        l = Dense(50, activation='relu')(l)
        l = BatchNormalization()(l)
        l = Dense(self.output_length, activation='linear')(l)

        model = Model(inputs=[img, meta], outputs=l)
        model.compile(loss='mae', optimizer='adam')
        return model

class ModelSimple_d(ModelBase):
    def build_model(self):

        img = Input(shape=self.input_shape)
        meta = Input(shape=(1,))

        model = Sequential()
        model = Sequential()

        model.add(img)
        model.add(Conv2D(24, kernel_size=(3, 3),
                         activation='relu',
                         data_format="channels_last"))
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())

        extra = Sequential()
        extra.add(Activation('linear', input_shape=(1,)))

        model.add(Concatenate([convnet, extra]))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.output_length, activation='linear'))

        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        return model

class ModelSimple_good(ModelBase):
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape,
                         data_format="channels_last"))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.output_length, activation='linear'))

        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        return model

class ModelNVIDIA(ModelBase):
    def build_model(self):
        print('NVIDIA model used')
        # model = Sequential()
        # model.add(Conv2D(24, kernel_size=(5, 5),
        #                  strides=(2, 2),
        #                  activation='relu',
        #                  input_shape=self.input_shape,
        #                  data_format="channels_last"))
        # model.add(Conv2D(36, kernel_size=(5, 5),
        #                  strides=(2, 2),
        #                  activation='relu'))
        # model.add(Conv2D(48, kernel_size=(5, 5),
        #                  strides=(2, 2),
        #                  activation='relu'))
        # model.add(Conv2D(64, kernel_size=(3, 3),
        #                  strides=(1, 1),
        #                  activation='relu'))
        # model.add(Conv2D(64, kernel_size=(3, 3),
        #                  strides=(1, 1),
        #                  activation='relu'))
        # model.add(Flatten())
        # model.add(Dense(100, activation='relu'))
        # model.add(Dense(10, activation='relu'))
        # model.add(Dense(self.output_length, activation='linear'))

        # model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        # return model

        img = Input(shape=self.input_shape)
        meta = Input(shape=(2,))

        l = Conv2D(24, kernel_size=(5, 5),
                         activation='relu',
                         data_format="channels_last")(img)
        l = BatchNormalization()(l)
        l = Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(l)
        l = BatchNormalization()(l)
        l = Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(l)
        l = BatchNormalization()(l)
        l = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(l)
        l = BatchNormalization()(l)
        l = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(l)
        l = Flatten()(l)

        l = keras.layers.concatenate([l, meta], axis=-1)

        l = Dense(100, activation='relu')(l)
        l = BatchNormalization()(l)
        l = Dense(10, activation='relu')(l)
        l = Dense(self.output_length, activation='linear')(l)

        model = Model(inputs=[img, meta], outputs=l)
        model.compile(loss='mae', optimizer='adam')
        return model

class ModelFromFile(ModelBase):
    def __init__(self, filename):
        self.filename = filename
        super(ModelFromFile, self).__init__((), 1)

    def build_model(self):
        json_file = open('{}.json'.format(self.filename), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        try:
            model.load_weights("{}.h5".format(self.filename))
            print('loaded file weights')
        except:
            model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        return model
