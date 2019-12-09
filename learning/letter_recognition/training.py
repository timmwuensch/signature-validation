import keras
import numpy as np
import idx2numpy
import cv2
import math
from contextlib import redirect_stdout
from models import best_model

# Define a callback class
class MySaver(keras.callbacks.Callback):
    def __init__(self, model):
        self.file = open('log_model_27_A.txt', 'a')
        with redirect_stdout(self.file):
            model.summary()


    def on_epoch_end(self, epoch, logs={}):
        self.file.write(str(logs) + "\n")

    def on_train_end(self, logs=None):
        self.file.close()

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
nb_classes = 27

# Load letter dataset
X_train = np.load('../data/letter/unbalanced/27/X_train.npy')
X_test = np.load('../data/letter/unbalanced/27/X_test.npy')
Y_train = np.load('../data/letter/unbalanced/27/Y_train.npy')
Y_test = np.load('../data/letter/unbalanced/27/Y_test.npy')

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Normalize values
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

# Data Augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.25,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.2)
datagen.fit(X_train)


# Define training parameters
epochs = 20
batch_size = 64

# Define class weights to handle unbalanced datasets
class_weights = {0: 0.03611801106898505, 1: 0.02263767231736901, 2: 0.06144884589125666, 3: 0.02655975363518833, 4: 0.029929838323092345, 5: 0.0030069290103281474, 6: 0.014941677198181316, 7: 0.018840516552635782, 8: 0.0029139611568687264, 9: 0.022265800903531325, 10: 0.01457271102976424, 11: 0.030345288418239132, 12: 0.03231795005883122, 13: 0.049641928501910196, 14: 0.15127612905100157, 15: 0.05062390145407533, 16: 0.015165381095568049, 17: 0.030153542220479074, 18: 0.12647985938612163, 19: 0.05875568338635406, 20: 0.07605060937522697, 21: 0.010859807382228614, 22: 0.028334858587179153, 23: 0.01633910024549324, 24: 0.028407489722694326, 25: 0.015882976714457955, 26: 0.026129777312938512}


model = best_model.Letter_Recognition_Model.load_model_A()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

saver = MySaver(model)
checkpoint = keras.callbacks.ModelCheckpoint('best_model_27_A.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=math.ceil(len(X_train) / batch_size),
                            epochs=epochs, validation_data=(X_test, Y_test), callbacks=[saver, checkpoint], class_weight=class_weights)

model.save('letter_recognition_model.h5')



