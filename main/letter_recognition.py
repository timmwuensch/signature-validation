## Letter Recognition Component

import cv2
import numpy as np
import keras
import image_processing as ip
import os


PATH = os.path.join('E:/', 'Programming', 'signature-validation', 'main', 'trained_models', 'letter_recognition_model.h5')
L_MODEL = keras.models.load_model(PATH)
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?"

def get_letters(signature):
    # Cut image to AOI and invert it
    borderless = ip.get_borderless_image(signature)
    borderless_black = 255 - borderless

    # Define segments of signature
    first_image, second_image = ip.get_samples_by_segments(borderless_black)

    # Build input for CNN
    letter_input = np.zeros((2, 28, 28))
    letter_input[0, :, :] = first_image
    letter_input[1, :, :] = second_image

    letter_input = np.expand_dims(letter_input, axis=3)
    letter_input = letter_input.astype('float32')
    letter_input /= 255

    # Make prediction with CNN
    prediction = L_MODEL.predict(letter_input)

    # Get letters of prediction by first choice
    first_letter1 = ALPHABET[np.argmax(prediction[0])]
    second_letter1 = ALPHABET[np.argmax(prediction[1])]

    # Get letters of prediction by second choice
    prediction[0][np.argmax(prediction[0])] = 0
    prediction[1][np.argmax(prediction[1])] = 0

    first_letter2 = ALPHABET[np.argmax(prediction[0])]
    second_letter2 = ALPHABET[np.argmax(prediction[1])]

    return (first_letter1, second_letter1, first_letter2, second_letter2)