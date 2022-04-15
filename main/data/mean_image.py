from PIL import Image
import os
import numpy as np
import sys
from data_loader import load_all_data
images_by_label= {
  0:[],
  1:[],
  2:[],
  3:[], 
  4:[],
  5:[],
  6:[],
  7:[], 
  8:[],
  9:[],
  10:[],
  11:[],
  12:[],
  13:[],
  14:[], 
  15:[],
  16:[], 
  17:[], 
  18:[], 
  19:[],
}
def mean():
    '''Compute the mean image for each number'''
    data=load_all_data()
    images=data[0][0]
    labels=data[0][1]
    assert(len(images)==len(labels))
    mean=np.zeros((max(labels),28,28))
    for i in range (len(images)):
        for j in range(20):
            if labels[i]==j:
                images_by_label[j].append(images[i])
    for k in range(max(labels)):
        for l in images_by_label[k]:
                mean[k]+=l/len(images_by_label[k])
    return mean
mean_arrays=mean()
i=0
for arrays in mean_arrays:
    y=Image.fromarray(arrays.astype(np.uint8))
    y.save('main\data\mean_images\mean_image_{}.png'.format(i),'png')
    i+=1
