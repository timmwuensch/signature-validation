import numpy as np
import tensorflow as tf
import keras
import grid_models
from contextlib import redirect_stdout
import math
import os
import random


# Set random seeds to make results more comparable
seed_value = 2019
os.environ['PYTHONHASHSEED']=str(seed_value)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



# Globat parameters
training = True
data_augmentation = True
dataset = 'E:/Programming/bi-thesis-tw/bi-thesis-tw/learning/data/format/small_dataset_grayscale/'
#dataset = 'E:/Programming/bi-thesis-tw/bi-thesis-tw/learning/data/format/small_dataset_RGB/'

classes = {0: "all other", 1: "F. Lastname"}


# Define a callback class to log results
class MySaver(keras.callbacks.Callback):
    def __init__(self, model, idx):
        self.model = model
        self.idx = idx
        self.file = open(dataset + str(self.idx) + '.txt', 'a')
        self.sum_file = open(dataset + 'summary' + '.txt', 'a')
        self.last_log = None


    def on_train_begin(self, logs=None):
        with redirect_stdout(self.file):
            self.model.summary()

    def on_epoch_end(self, epoch, logs={}):
        self.file.write(str(logs) + "\n")
        self.last_log = logs

    def on_train_end(self, logs=None):
        self.sum_file.write(str(self.last_log) + "\n")
        self.sum_file.close()
        self.file.close()



# Load predefined image dataset (signatures)
X_train = np.load(dataset + 'X_train.npy')
X_test = np.load(dataset + 'X_test.npy')
Y_train = np.load(dataset + 'Y_train.npy')
Y_test = np.load(dataset + 'Y_test.npy')

#X_train = np.expand_dims(X_train, axis=3)
#X_test = np.expand_dims(X_test, axis=3)

X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.astype('float32')
X_test /= 255


# Initialize data generator for data augmentation
if data_augmentation:
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)


# Define training parameters and generate models
epochs = 50
batch_size = 16

#models = grid_models.Format_Recognition_Model.generate_new_models()
models = grid_models.Format_Recognition_Model.generate_transfer_models()


for idx, model in enumerate(models):

    # Reset random seeds
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    saver = MySaver(model, idx)

    if training and data_augmentation:
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=math.ceil(len(X_train) / batch_size),
                            epochs=epochs, validation_data=(X_test, Y_test),
                            callbacks=[saver])

    elif training and not data_augmentation:
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test),
                            callbacks=[saver])

    del model
    del saver
