import numpy as np

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
def move_image(sample):
    new_images = []
    right = moving_right(sample)
    left = moving_left(sample)
    up = moving_up(sample)
    down = moving_down(sample)

    if len(right) != 0:
        new_images.append(right)

    if len(left) != 0:
        new_images.append(left)

    if len(up) != 0:
        new_images.append(up)

    if len(down) != 0:
        new_images.append(down)

    return new_images

def moving_right(sample):
    all_zeros = True
    for row in sample:
        if row[len(sample) - 1] != 0:
            all_zeros = False
            break

    if all_zeros:
        new_sample = []
        for row in sample:
            new_sample.append(np.concatenate(([0], row[0:len(sample) - 1])))
        return new_sample
    return []

def moving_left(sample):
    all_zeros = True
    for row in sample:
        if row[0] != 0:
            all_zeros = False
            break

    if all_zeros:
        new_sample = []
        for row in sample:
            new_sample.append(np.concatenate((row[1:], [0])))
        return new_sample
    return []

def moving_up(sample):
    if not all(sample[0]):
        return np.concatenate((sample[1:len(sample)], [np.zeros(28)]))
    return []

def moving_down(sample):
    if not all(sample[len(sample) - 1]):
        return np.concatenate(([np.zeros(28)], sample[:27]))
    return []
