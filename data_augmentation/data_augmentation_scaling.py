from typing import final
from PIL import Image
import os
import numpy as np
import skimage
import cv2


def zoom(path,enlargement):
    img = cv2.imread(path)

    width = int(img.shape[1] * enlargement)
    height = int(img.shape[0] * enlargement)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    if enlargement > 1:
        final_img = resized[np.abs(28 - width)//2:np.abs(28 + height)//2, np.abs(28 - width)//2:np.abs(28 + height)//2]

    else:
        final_img = np.zeros((28, 28, 3))
        final_img[np.abs(28 - width)//2:np.abs(28 + height)//2, np.abs(28 - width)//2:np.abs(28 + height)//2] = resized

    return(final_img)

cv2.imshow("sample", zoom('./data/sylheti/0_0.png',0.5))
#cv2.imshow("sample1", img)
cv2.waitKey(0)
