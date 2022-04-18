import cv2

# Quick function that can take an angle and rotate that image by 
# that many degrees
def rotate_image(image, angle):
    rot_mat = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (28, 28))
