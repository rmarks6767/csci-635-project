import numpy as np
import cv2

def scale_image(img, scale):
    size = int(28 * scale)
    dim = (size, size)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    if scale > 1:
        final_img = resized[np.abs(28 - size)//2:np.abs(28 + size)//2, np.abs(28 - size)//2:np.abs(28 + size)//2]

    else:
        final_img = np.zeros((28, 28))
        final_img[np.abs(28 - size)//2:np.abs(28 + size)//2, np.abs(28 - size)//2:np.abs(28 + size)//2] = resized

    return(final_img)

