from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Activation, AveragePooling2D

class Letter_Recognition_Model():

    @staticmethod
    def load_model_A():
        model = Sequential()

        model.add(Convolution2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu', padding="same"))
        model.add(Convolution2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
        model.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(26, activation='softmax'))

        return model

    @staticmethod
    def load_model_lenet():
        model = Sequential()

        model.add(Convolution2D(6, (3, 3), input_shape=(28, 28, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(16, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(27, activation='softmax'))

        return model
