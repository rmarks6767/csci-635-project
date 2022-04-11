import numpy as np
import os

'''
  Data is saved as an n x (28 x 28) numpy array. If the data should be flattened, 
  pass in the flatten parameter
'''
def load_data(lang, flatten = False):
  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'./{lang}/{lang}_train_images.npz'))
  train_images = train_images = npzfile['arr_0']

  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'./{lang}/{lang}_train_labels.npz'))
  train_labels = npzfile['arr_0']

  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'./{lang}/{lang}_test_images.npz'))
  test_images = npzfile['arr_0']

  npzfile = np.load(os.path.join(os.path.dirname(__file__), f'./{lang}/{lang}_test_labels.npz'))
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

  # Return the training and test data
  return (train_images, train_labels), (test_images, test_labels)
