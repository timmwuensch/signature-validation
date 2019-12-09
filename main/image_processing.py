## The following methods are used to support the single components with image processing functions

import numpy as np
import cv2

def show_image(image, title='Test'):
    ## Method shows an image
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_28_square_image(image):
    ## Method creates a 28x28 square image by rescaling the given original image and pads it with zeros
    h,w = image.shape
    dif = h-w

    if dif == 0:
        square = cv2.resize(image, (28, 28))

    if dif > 0:
        square = np.zeros((h, h))
        pos = abs(int(dif/2)-1)
        square[0:h, pos:(pos+w)] = image

        square = cv2.resize(square, (28,28))

    elif dif < 0:
        square = np.zeros((w, w))
        pos = abs(int(dif/2)-1)
        square[pos:(pos+h), 0:w] = image

        square = cv2.resize(square, (28, 28))

    return square

def get_borderless_image(signature):
    ## Method reduces the image to region of interest
    img = signature

    x,y = find_upper_left_corner(img)
    xx,yy =find_lower_right_corner(img)

    #cv2.rectangle(signature, (y, x), (yy, xx), (255, 0, 0), 2)
    new = signature[x:xx, y:yy]

    return new


def find_upper_left_corner(image):
    ## Helper function for get_borderless_image
    h,w = image.shape
    x = 0
    y = 0

    for idx in range(0,h):
        if x > 0:
            break
        for idy in range(0,w):
            val = image[idx][idy]
            if val < 200 :
                x = idx
                break

    for idy in range(0,w):
        if y > 0:
            break
        for idx in range(0,h):
            val = image[idx][idy]
            if val <200 :
                y = idy
                break

    return (x,y)

def find_lower_right_corner(image):
    ## Helper function for get_borderless_image
    h, w = image.shape
    xx = h-1
    yy = w-1

    for idx in range(h-1, 0, -1):
        if xx < (h-1):
            break
        for idy in range(w-1, 0, -1):
            val = image[idx][idy]
            if val < 200:
                xx = idx
                break

    for idy in range(w-1, 0, -1):
        if yy < (w-1):
            break
        for idx in range(h-1, 0, -1):
            val = image[idx][idy]
            if val < 200:
                yy = idy
                break

    return (xx, yy)

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Source: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    # initialize the dimensions of the image to be resized and

    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def get_segments(image):
    # This function returns a list of segments of letters

    transposed = np.transpose(image)
    img_row_sum = np.sum(transposed, axis=1).tolist()

    img_row_sum = [1 if i>0 else 0 for i in img_row_sum]
    segments = []
    sequence = []
    temp = []
    mode = 1

    for idx, val in enumerate(img_row_sum):
        if mode == val:
            temp.append(idx)
        else:
            if len(temp) == 1: sequence.append((mode, temp[0], temp[0]))
            elif len(temp) == 0: pass
            else: sequence.append((mode, temp[0], temp[-1]))
            temp = [idx]
            mode = val
    if len(temp) > 1:
        sequence.append((mode, temp[0], temp[-1]))

    idx_start = 0
    for idx, seg in enumerate(sequence):
        val, start, end = seg
        if val == 0 and (end-start) >= 2:
            segments.append((idx_start, start-1))
            idx_start = end+1
    if (idx_start < sequence[-1][2]-1):
        segments.append((idx_start, sequence[-1][2]-1))

    return segments

def get_samples_by_segments(image):
    # This function returns the first and second letter as an 28x28 image tuple

    segments = get_segments(image)
    h,w = image.shape
    segments = [i for i in segments if (i[1] - i[0]) > 10]

    if len(segments) <= 1:      # Wenn es nur ein oder kein Segment gibt
        first_segment = (0, 88)
        second_segment = (90, 178)

    elif len(segments) == 2:
        start = segments[0][0]
        end = segments[0][1]

        if (end - start) > 150: # Wenn das erste Segment zu groß ist
            first_segment = (0, 88)
            second_segment = (90, 178)
        else:
            first_segment = segments[0] # Wenn das zweite Segment zu groß ist
            second_segment = (end, end + 88)

    else:
        first_segment = segments[0]

        start = segments[1][0]
        end = segments[1][1]

        if (end - start) > 88:
            second_segment = (start, start + (first_segment[1]-first_segment[0]))
        else:
            second_segment = (start, end)

    first_letter = image[0:h, first_segment[0]:first_segment[1]]
    second_letter = image[0:h, second_segment[0]:second_segment[1]]

    first_letter = make_28_square_image(first_letter)
    second_letter = make_28_square_image(second_letter)

    return (first_letter, second_letter)