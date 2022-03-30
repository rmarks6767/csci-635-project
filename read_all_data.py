import numpy as np
from numpy import genfromtxt
from mnist import MNIST
import os
from skimage.io import imread
import numpy as np
from random import sample

'''
  Helper function to read all of the data into one giant array so
  we can run it through our training algorithm
'''
def read_all_data():
  # # Get the data and labels for the arabic numerals
  # print('READING ARABIC DATA')
  # arabic_train_data = genfromtxt("./data/arabic/csvTrainImages 60k x 784.csv", delimiter=',')
  # arabic_test_data = genfromtxt("./data/arabic/csvTestImages 10k x 784.csv", delimiter=',')
  # arabic_train_label = genfromtxt("./data/arabic/csvTrainLabel 60k x 1.csv", delimiter=',')
  # arabic_test_label = genfromtxt("./data/arabic/csvTestLabel 10k x 1.csv", delimiter=',')

  # # We will split these apart later
  # arabic_data = np.concatenate((arabic_train_data, arabic_test_data))
  # arabic_label = np.concatenate((arabic_train_label, arabic_test_label))

  # print('FINISHED READING ARABIC DATA')

  # Get the data and labels for the mnist
  print('READING THE ENGLISH DATA')
  mndata = MNIST(os.path.join(os.path.dirname(__file__), 'data/english'))
  # english_train_data, english_train_label = mndata.load_training()
  english_test_data, english_test_label = mndata.load_testing()

  # english_data = np.concatenate((english_train_data, english_test_data))
  # english_label = np.concatenate((english_train_label, english_test_label))

  english_data = english_test_data
  english_label = english_test_label

  print('FINISHED READING ENGLISH DATA')

  # Get the data and labels for the Sylheti
  print('READING THE SYLHETI DATA')
  sylheti_data = []
  sylheti_label = []

  for img in os.listdir('./data/sylheti'):
    # Read in the images
    _, label = img.replace('.png', '').split('_')
    image = imread(os.path.join('data/sylheti', img), as_gray=True)

    # Append them to our images and labels array
    sylheti_data.append(image.flatten())
    sylheti_label.append(label)

  print('FINISHED READING SYLHETI DATA')

  # # Combine all of them into one data source
  # all_data = np.concatenate((arabic_data, english_data, sylheti_data))
  # all_label = np.concatenate((arabic_label, english_label, sylheti_label))

  all_data =  all_data = np.concatenate((english_data, sylheti_data))
  all_label = np.concatenate((english_label, sylheti_label))


  print(f'TOTAL EXAMPLES LOADED: {len(all_data)}')

  return all_data, all_label
