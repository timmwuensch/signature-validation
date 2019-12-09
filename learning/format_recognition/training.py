## This Python script trains on explicit model over 300 epochs on a big dataset.

import numpy as np
import tensorflow as tf
import keras
from models import best_model
from contextlib import redirect_stdout
import math
import os
import random


# Random Seed
seed_value = 2019
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# Global parameters
training = True
data_augmentation = True
dataset = 'E:/Programming/bi-thesis-tw/bi-thesis-tw/learning/data/format/big_dataset_grayscale/'

classes = {0: "all other", 1: "F. Lastname"}

# Define a callback class to log results
class MySaver(keras.callbacks.Callback):
    def __init__(self, model):
        self.file = open(dataset + 'A24_adam_16_big.txt', 'a')
        with redirect_stdout(self.file):
            model.summary()

    def on_epoch_end(self, epoch, logs={}):
        self.file.write(str(logs) + "\n")

    def on_train_end(self, logs=None):
        self.file.close()

# Load dataset
X_train = np.load(dataset + 'X_train.npy')
X_test = np.load(dataset + 'X_test.npy')
Y_train = np.load(dataset + 'Y_train.npy')
Y_test = np.load(dataset + 'Y_test.npy')

# Expand dimension to handle grayscale images
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Normalize pixel values
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255


# Define training parameters
epochs = 300
batch_size = 16


# Initialize data generator for data augmentation
if data_augmentation:
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(X_train)


# Load model
model = best_model.Format_Recognition_Model.load_model_A24()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

saver = MySaver(model)
checkpoint = keras.callbacks.ModelCheckpoint('best_A24_adam_16_big.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)

# Reset random seeds
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

if training and data_augmentation:
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=math.ceil(len(X_train) / batch_size),
                        epochs=epochs, validation_data=(X_test, Y_test),
                        callbacks=[saver, checkpoint])

elif training and not data_augmentation:
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test),
                        callbacks=[saver])

path = dataset + 'A24_adam_16_big.h5'
model.save(path)