## This is the main programm of the project.
## It is used by calling the method validate signature with the parameters signature and name.
## The method gives back a boolean value with some print-outs.

import sys
import numpy as np
import cv2
import keras
import glob

import image_processing as ip
import format_recognition
import letter_recognition


def validate_signature(image, name):

    # Call the format recognition component
    format = format_recognition.get_format(image)

    # Check if format recognition was successfull
    if format == 0:
        print("Validation negative: Format is not valid.")
        return False

    # Get letters from letter recognition
    (first_letter1, second_letter1, first_letter2, second_letter2) = letter_recognition.get_letters(image)

    # Find beginning letters from full name
    words = name.split(" ")
    words = [i.capitalize() for i in words]
    letters = [word[0] for word in words]

    # Check occurrence of predicted letters in list of beginning letters
    if (first_letter1 or first_letter2 or second_letter1 or second_letter2) in letters:
        print("Validation positive")
        return True

    # Return successfull validation
    print("Validation negative: Letters could not be recognized.")
    return False


