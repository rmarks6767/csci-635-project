import cv2

def rotate_image(image, angle):
    rot_mat = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (28, 28))
