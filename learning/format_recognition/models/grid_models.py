## This Python script defines the models which are examined in the grid search.
## Via class Format_Recognition_Model it is possible to generate new models and transfered models (VGG16, VGG19)

import keras
from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Activation, AveragePooling2D


# Helper functions to create sequential keras models

def add_input_conv_layer(model, num_filter):
    model.add(Convolution2D(num_filter, kernel_size=3, activation='relu', padding='same', input_shape=(60, 200, 1)))
    return model

def add_conv_layer(model, num_filter):
    model.add(Convolution2D(num_filter, kernel_size=3, activation='relu'))
    return model

def add_pooling_layer(model):
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model

def add_flatten_layer(model):
    model.add(Flatten())
    return model

def add_dropout_layer(model, dropout):
    model.add(Dropout(dropout))
    return model

def add_dense_layer(model, num_neurons):
    model.add(Dense(num_neurons, activation='relu'))
    return model

def add_output_layer(model):
    model.add(Dense(2, activation='softmax'))
    return model



# Main class for model generation

class Format_Recognition_Model():

    @staticmethod
    def generate_transfer_models():
        dense_parts = [[0.3, 256, 0.3, 64, 0.3],
                       [0.3, 256, 0.3, 128, 0.3],
                       [0.3, 512, 0.3, 64, 0.3],
                       [0.3, 512, 0.3, 128, 0.3],
                       [0.5, 256, 0.5, 64, 0.5],
                       [0.5, 256, 0.5, 128, 0.5],
                       [0.5, 512, 0.5, 64, 0.5],
                       [0.5, 512, 0.5, 128, 0.5]]

        pretrained = []
        conv_parts = []
        models = []

        vgg16 = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(60, 200, 3))
        vgg19 = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(60, 200, 3))

        for layer in vgg16.layers:
            layer.trainable = False

        for layer in vgg19.layers:
            layer.trainable = False

        pretrained.append(vgg16)
        pretrained.append(vgg19)

        for vgg in pretrained:
            conv = Sequential()
            conv.add(vgg)
            conv.add(Flatten())
            conv_parts.append(conv)

        for conv_part in conv_parts:
            for dense_part in dense_parts:
                model = keras.models.clone_model(conv_part)
                model.set_weights(conv_part.get_weights())

                model = add_dropout_layer(model, dense_part[0])
                model = add_dense_layer(model, dense_part[1])
                model = add_dropout_layer(model, dense_part[2])
                model = add_dense_layer(model, dense_part[3])
                model = add_dropout_layer(model, dense_part[4])
                model = add_output_layer(model)

                models.append(model)

        return models


    @staticmethod
    def generate_new_models():
        convs = [[8, 8],
                 [8, 8, 16],
                 [8, 32],
                 [8, 32, 16],
                 [16, 8],
                 [16, 8, 16],
                 [16, 32],
                 [16, 32, 16],]

        dense_parts = [[0.3, 256, 0.3, 64, 0.3],
                       [0.3, 256, 0.3, 128, 0.3],
                       [0.3, 512, 0.3, 64, 0.3],
                       [0.3, 512, 0.3, 128, 0.3],
                       [0.5, 256, 0.5, 64, 0.5],
                       [0.5, 256, 0.5, 128, 0.5],
                       [0.5, 512, 0.5, 64, 0.5],
                       [0.5, 512, 0.5, 128, 0.5]]

        models = []
        conv_parts = []

        for conv_seq in convs:
            model = Sequential()

            model = add_input_conv_layer(model, conv_seq[0])
            model = add_pooling_layer(model)
            for conv in conv_seq[1:]:
                model = add_conv_layer(model, conv)
                model = add_pooling_layer(model)

            model = add_flatten_layer(model)
            conv_parts.append(model)

        for conv_part in conv_parts:
            for dense_part in dense_parts:
                model = keras.models.clone_model(conv_part)
                model = add_dropout_layer(model, dense_part[0])
                model = add_dense_layer(model, dense_part[1])
                model = add_dropout_layer(model, dense_part[2])
                model = add_dense_layer(model, dense_part[3])
                model = add_dropout_layer(model, dense_part[4])
                model = add_output_layer(model)
                models.append(model)

        return models


