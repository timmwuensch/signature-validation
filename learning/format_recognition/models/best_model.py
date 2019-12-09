## This Python script defines the sequential structure of the best examined model.
## See documentation for results.

import keras
from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Activation, AveragePooling2D

class Format_Recognition_Model():

    @staticmethod
    def load_model_A11():
        model = Sequential()

        model.add(Convolution2D(8, kernel_size=3, activation='relu', padding='same', input_shape=(60, 200, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(8, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(16, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(2, activation='softmax'))

        return model