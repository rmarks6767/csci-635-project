import tensorflow as tf
import numpy as np
import os

labels = {
  0: 'eng-syl-0/arb-5',
  1: 'all-1',
  2: 'eng-2',
  3: 'eng-3',
  4: 'eng-4',
  5: 'eng-5',
  6: 'eng-6',
  7: 'eng-7/arb-6',
  8: 'eng-syl-8',
  9: 'all-9',
  10: 'arb-0',
  11: 'arb-syl-2',
  12: 'arb-3',
  13: 'syl-3',
  14: 'arb-4',
  15: 'syl-4',
  16: 'syl-5',
  17: 'syl-6',
  18: 'arb-syl-7',
  19: 'arb-8',
}

'''
  Data is saved as an n x (28 x 28) numpy array. If the data should be flattened, 
  pass in the flatten parameter
'''
def load_data(lang, flatten = False):
  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'../data/{lang}/{lang}_train_images.npz'))
  train_images = train_images = npzfile['arr_0']

  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'../data/{lang}/{lang}_train_labels.npz'))
  train_labels = npzfile['arr_0']

  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'../data/{lang}/{lang}_test_images.npz'))
  test_images = npzfile['arr_0']

  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'../data/{lang}/{lang}_test_labels.npz'))
  test_labels = npzfile['arr_0']

  if flatten:
    return (
      train_images.flatten().reshape(len(train_images), 784), 
      train_labels
    ),(
      test_images.flatten().reshape(len(test_images), 784), 
      test_labels
    )
  return (train_images, train_labels), (test_images, test_labels)

def load_all_data(flatten = False):
  # Load each of the langauges
  (arb_train_imgs, arb_train_lbls), (arb_test_imgs, arb_test_lbls) = load_data('arabic', flatten)
  (eng_train_imgs, eng_train_lbls), (eng_test_imgs, eng_test_lbls) = load_data('english', flatten)
  (syl_train_imgs, syl_train_lbls), (syl_test_imgs, syl_test_lbls) = load_data('sylheti',flatten)

  # Combine the training images and labels
  train_images = np.concatenate((arb_train_imgs, eng_train_imgs, syl_train_imgs))
  train_labels = np.concatenate((arb_train_lbls, eng_train_lbls, syl_train_lbls))

  # Combine the test images and labels
  test_images = np.concatenate((arb_test_imgs, eng_test_imgs, syl_test_imgs))
  test_labels = np.concatenate((arb_test_lbls, eng_test_lbls, syl_test_lbls))

  # # One hot encode the labels
  # train_labels = tf.one_hot(train_labels, 20)
  # test_labels = tf.one_hot(test_labels, 20)

  print(f'Total Training images loaded: {len(train_images)}')
  print(f'Total Test images loaded: {len(test_images)}')

  # Return the training and test data
  return (train_images, train_labels), (test_images, test_labels)
