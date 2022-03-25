import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mnist import MNIST
import os
from skimage.io import imread
import numpy as np
import random

def read_all_data():
  # Get the data and labels for the arabic numerals
  arabic_train_data = genfromtxt("./data/arabic/csvTrainImages 60k x 784.csv", delimiter=',')
  arabic_test_data = genfromtxt("./data/arabic/csvTestImages 10k x 784.csv", delimiter=',')
  arabic_train_label = genfromtxt("./data/arabic/csvTrainLabel 60k x 1.csv", delimiter=',')
  arabic_test_label = genfromtxt("./data/arabic/csvTestLabel 10k x 1.csv", delimiter=',')

  print(len(arabic_train_data))
  print(len(arabic_train_data[0]))

  # We will split these apart later
  arabic_data = np.concatenate((arabic_train_data, arabic_test_data))
  arabic_label = np.concatenate((arabic_train_label, arabic_test_label))

  # Get the data and labels for the mnist
  mndata = MNIST(os.path.join(os.path.dirname(__file__), 'data/english'))
  english_data, english_label = mndata.load_training()
  english_data = np.array(english_data)
  english_label = np.array(english_label)


  index = random.randrange(0, len(english_data))  # choose an index ;-)
  print(mndata.display(english_data[index]))

  # Get the data and labels for the Sylheti
  sylheti_data = []
  sylheti_label = []

  for img in os.listdir('./data/sylheti'):
    # Read in the images
    _, label = img.replace('.png', '').split('_')
    image = imread(os.path.join('data/sylheti', img))

    # Resize it to be the appropriate size (should already be this for ours)
    # im_resized = resize(image, (28, 28))

    # Append them to our images and labels array
    sylheti_data.append(image)
    sylheti_label.append(label)
  
  print(len(arabic_data))
  print(len(arabic_data[0]))
  print(len(english_data))
  print(len(english_data[0]))
  # print(len(sylheti_data))
  # print(len(sylheti_data[0]))

  # Combine all of them into one data source
  all_data = np.concatenate((arabic_data, english_data, sylheti_data))
  all_label = np.concatenate((arabic_label, english_label, sylheti_label))

  return np.array(all_data), np.array(all_label)

read_all_data()