import models

import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class ModelToSave(models.ModelBase):
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(24, kernel_size=(5, 5),
                         strides=(1, 1),
                         activation='relu',
                         input_shape=self.input_shape,
                         data_format="channels_last"))
        model.add(Conv2D(36, (5, 5), 
                         activation='relu',
                         strides=(2, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(48, (5, 5), 
                         activation='relu',
                         strides=(2, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), 
                         activation='relu',
                         strides=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.output_length, activation='linear'))

        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        return model

input_shape = (120, 160, 1)
output_length = 1
m = ModelToSave(input_shape, output_length)
m.save('models/nvidia_dropout_last_50', save_weights=False)
