# This augmentation will move the data inside around while still keeps the general feature of the data
import os
from skimage.io import imread
import numpy as np
from mnist import MNIST


# To make printing look nicer
np.set_printoptions(linewidth=np.inf)


#############################################################
# This is the main function to call to get augmented data
#
# INPUT: An array that represent the number
#        (For example, grayscale values of each pixel in the picture of the number)
#
# RETURN: A list that has 4 new arrays:
#     1. The number is moved right 1 pixel
#     2. The number is moved left 1 pixel
#     3. The number is moved up 1 pixel
#     4. The number is moved down 1 pixel
#############################################################
def data_augment(sample):
    # the sample picture is 28x28
    size = 28
    # The result of augmenting the data
    # will be empty if the sample cannot be augmented
    result = [moving_right(sample, size), moving_left(sample, size),
              moving_up(sample, size), moving_down(sample, size)]
    return result


def moving_right(sample, size):
    # Moving the entire object to the right

    # Objects in the images will have a value less than 1
    # for every pixel that they are in
    # 1 means nothing is there
    zero = True
    idx = size - 1
    # Check if the rightmost edge of the sample is empty
    while zero and idx < size*size-1:
        if sample[idx] != 1:
            zero = False
        else:
            idx = idx + size
    if zero:
        idx = 0
        new_sample = np.asarray([])
        while idx < size*size-1:
            new_sample = np.concatenate((new_sample, np.ones(1), sample[idx:(idx+size-1)]))
            idx = idx + size
        return new_sample
    else:
        return []


def moving_left(sample, size):
    # Moving the entire object to the left

    # Objects in the images will have a value less than 1
    # for every pixel that they are in
    # 1 means nothing is there
    zero = True
    idx = 0
    # Check if the leftmost edge of the sample is empty
    while zero and idx < len(sample):
        if sample[idx] != 1:
            zero = False
        else:
            idx = idx + size
    if zero:
        idx = 1
        new_sample = np.asarray([])
        while idx < len(sample):
            new_sample = np.concatenate((new_sample, sample[idx:(idx+size-1)], np.ones(1)))
            idx = idx + size
        return new_sample
    else:
        return []


def moving_up(sample, size):
    # Moving the entire object up
    # Objects in the images will have a value less than 1
    # for every pixel that they are in
    # 1 means nothing is there
    zero = True
    idx = 0
    # Check if the topmost edge of the sample is empty
    while zero and idx < size-1:
        if sample[idx] != 1:
            zero = False
        else:
            idx = idx + 1
    if zero:
        new_sample = np.concatenate((sample[size:len(sample)], np.ones(size)))
        return new_sample
    else:
        return []


def moving_down(sample, size):
    # Moving the entire object down
    # Objects in the images will have a value less than 1
    # for every pixel that they are in
    # 1 means nothing is there
    zero = True
    idx = size*size - size
    # Check if the bottommost edge of the sample is empty
    while zero and idx < len(sample):
        if sample[idx] != 1:
            zero = False
        else:
            idx = idx + 1
    if zero:
        idx = size*size-size
        new_sample = np.concatenate((np.ones(size), sample[0:idx]))
        return new_sample
    else:
        return []


def pretty_print(sample):
    idx = 0
    while idx < len(sample):
        print(sample[idx:(idx+28)])
        idx = idx + 28


def binarize(sample):
    result = []
    idx = 0
    while idx < len(sample):
        if sample[idx] != 1:
            result.append('.')
        else:
            result.append('0')
        idx += 1
    return result


######################
# Mainly for testing
######################
def main():
    sylheti_data = []
    sylheti_label = []
    for img in os.listdir('../data/sylheti'):
        # Read in the images
        _, label = img.replace('.png', '').split('_')
        image = imread(os.path.join('../data/sylheti', img), as_gray=True)

        # Append them to our images and labels array
        sylheti_data.append(image.flatten())
        sylheti_label.append(label)
    mndata = MNIST(os.path.join(os.path.dirname(__file__), '../data/english'))
    english_train_data, english_train_label = mndata.load_training()
    english_test_data, english_test_label = mndata.load_testing()

    english_data = np.concatenate((english_train_data, english_test_data))
    english_label = np.concatenate((english_train_label, english_test_label))
    new = binarize(sylheti_data[0])
    #pretty_print(sylheti_data[0])
    print(len(sylheti_data[0]))
    pretty_print(new)
    augmented = moving_right(sylheti_data[0], 28)
    while len(augmented) != 0:
        print(len(augmented))
        pretty_print(binarize(augmented))
        augmented = moving_right(augmented, 28)


main()