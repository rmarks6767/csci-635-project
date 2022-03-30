from PIL import Image
import os
import numpy as np
import skimage
import cv2


img = cv2.imread('./data/sylheti/0_0.png')

width = int(img.shape[1] * 1.5)
height = int(img.shape[0] * 1.5)
dim = (width, height)


resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


# # Scikit Image. 
# img = Image.open('./data/sylheti/0_0.png')

# img_img = Image.fromarray(img.astype(np.uint8))
# img_img.show()

# #print(np.shape(img))

# # For details about 'mode', checkout the interpolation section.
# scale_out = skimage.transform.rescale(img, scale=1.1, mode='constant')
# scale_in = skimage.transform.rescale(img, scale=0.9, mode='constant')
# # Don't forget to crop the images back to the original size (for scale_out)

# #print(np.shape(scale_out))
# #print(np.shape(scale_in))

# scale_out_img = Image.fromarray(scale_out.astype(np.uint8))
# #scale_in_img = Image.fromarray(scale_in.astype(np.uint8))

# scale_out_img.show()
# #scale_in_img.show()

# #print(scale_out_img)
# #print(scale_in)