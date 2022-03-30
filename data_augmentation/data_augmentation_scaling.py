from PIL import Image
import os
import numpy as np
import skimage
import cv2

# Scikit Image. 
img = Image.open('C:/Users/fabie/OneDrive/Documents/GitHub/sylheti-training-data/0-0.png')

#img_img = Image.fromarray(img.astype(np.uint8))
#img_img.show()

#print(np.shape(img))

# For details about 'mode', checkout the interpolation section.
scale_out = skimage.transform.rescale(img, scale=1.1, mode='constant')
scale_in = skimage.transform.rescale(img, scale=0.9, mode='constant')
# Don't forget to crop the images back to the original size (for scale_out)

#print(np.shape(scale_out))
#print(np.shape(scale_in))

scale_out_img = Image.fromarray(scale_out.astype(np.uint8))
#scale_in_img = Image.fromarray(scale_in.astype(np.uint8))

scale_out_img.show()
#scale_in_img.show()

#print(scale_out_img)
#print(scale_in)