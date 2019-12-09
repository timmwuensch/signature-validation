## Format Recognition Component

import cv2
import numpy as np
import keras
import image_processing as ip
import os

# Import model for Format Recognition
PATH = os.path.join('E:/', 'Programming', 'signature-validation', 'main', 'trained_models', 'format_recognition_model.h5')
F_MODEL = keras.models.load_model(PATH)


def get_format(signature):

    # Resize image into boundaries (60,200)
    if signature.shape[0] > 60: signature = ip.resize_image(signature, height=60)
    if signature.shape[1] > 200: signature = ip.resize_image(signature, width=200)

    # Build input for CNN
    h, w = signature.shape
    format_input = np.zeros((1, 60, 200))
    format_input[0, 0:h, 0:w] = signature

    format_input = np.expand_dims(format_input, axis=3)
    format_input = format_input.astype('float32')
    format_input /= 255

    # Make Prediction
    prediction = F_MODEL.predict(format_input)

    # Get Format
    format = np.argmax(prediction[0])

    return format