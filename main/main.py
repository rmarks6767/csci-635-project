from sklearn.model_selection import train_test_split
from data.data_loader import load_data
from augmentation.moving import move_image
from augmentation.rotation import rotate_image
from augmentation.scaling import scale_image
from models.svm import SVM
from models.cnn import CNN
from models.nn import NN
from models.transfer_nn import TransferNN
import argparse as ap
import numpy as np
import random
import time

# Helper function used to call all of our augmentors and 
# return all of the new data
def augment_data(images, labels):
  extra_images = []
  extra_labels = []
  i = 0
  for img in images:
    start_size = len(extra_images)

    # Move the images up, down, left and right
    extra_images.extend( move_image(img))

    # Randomly rotate the image
    extra_images.append(rotate_image(img, random.randint(15, 20)))

    # Randomly scale the image
    scale_factor = random.uniform(0.75, 1.15)
    extra_images.append(scale_image(img, scale_factor if scale_factor != 1 else 1.05))

    # Add the labels for the items that were found
    for _ in range(0, len(extra_images) - start_size):
      extra_labels.append(labels[i])

    i += 1

  return (np.array(extra_images), np.array(extra_labels))

def main():
  parser = ap.ArgumentParser(description='Running CSCI 635 Project')
  parser.add_argument('--training-only', help='Only run the model training on the project', action='store_true')
  parser.add_argument('--use-local-models', help='Use the locally trained models that have been provided', action='store_true')
  parser.add_argument('--no-verbose', help='Turns off the verbosity of Tensorflow', action='store_false')

  args = parser.parse_args()

  ############ Step 1 - Reading our data from compressed files ############ 
  (a_train_images, a_train_labels), (a_test_images, a_test_labels) = load_data('arabic')
  (e_train_images, e_train_labels), (e_test_images, e_test_labels) = load_data('english')
  (s_train_images, s_train_labels), (s_test_images, s_test_labels) = load_data('sylheti')
  
  ############ Step 2 - Augmenting our Sylheti data ############ 
  images = np.concatenate((s_train_images, s_test_images))
  labels = np.concatenate((s_train_labels, s_test_labels))
  (extra_images, extra_labels) = augment_data(images, labels)

  
  sylheti_images = np.concatenate((images, extra_images))
  sylheti_labels = np.concatenate((labels, extra_labels))

  ############ Step 2.1 - Re-split our data into test and train ############ 
  s_train_images, s_test_images, s_train_labels, s_test_labels = train_test_split(sylheti_images, sylheti_labels, test_size=0.20, random_state=42)

  ############ Step 2.2 - Combine our data to be run in the models ############ 
  # SVM Needs less data because it does not allow use of the GPU to train, we will test on the full corpus
  training_images_svm = np.concatenate((e_train_images[:3000], a_train_images[:3000], s_train_images[:3000]))
  training_labels_svm = np.concatenate((e_train_labels[:3000], a_train_labels[:3000], s_train_labels[:3000]))

  # Transfer learning should not include sylheti (that is the one we are transfering to)
  training_images_transfer = np.concatenate((e_train_images, a_train_images))
  training_labels_transfer = np.concatenate((e_train_labels, a_train_labels))

  # The rest will use all of the data
  training_images = np.concatenate((e_train_images, a_train_images, s_train_images))
  training_labels = np.concatenate((e_train_labels, a_train_labels, s_train_labels))
  test_images = np.concatenate((e_test_images, a_test_images, s_test_images))
  test_labels = np.concatenate((e_test_labels, a_test_labels, s_test_labels))

  ############ Step 3 - Model training ############ 
  # Create all models that we are comparing
  svm = SVM()
  cnn = CNN()
  nn = NN()
  cnn_transfer = CNN(model_filename='transfer_cnn_base.h5')
  transferNN = TransferNN(base_model=cnn_transfer)

  if not args.use_local_models:
    start = time.time()

    # Train all of our models 
    svm.train(training_images_svm, training_labels_svm)
    cnn.train(training_images, training_labels, args.no_verbose)
    nn.train(training_images, training_labels, args.no_verbose)

    # We have to also train the base model for our transferred model
    cnn_transfer.train(training_images_transfer, training_labels_transfer, args.no_verbose)
    transferNN.train(s_train_images, s_train_labels, args.no_verbose)

    print(f'\nTRAINING MODELS FINISHED IN: {round(time.time() - start, 2)} SECONDS')

  if not args.training_only:
    print('\n==================== BEGIN RESULTS ====================\n')

    ############ Step 4 - Gather statistics about our data on the model ############ 
    print('\n===== BEGIN RESULTS FOR SVM =====\n')
    svm.evaluate(test_images, test_labels, args.use_local_models)
    print('===== END RESULTS FOR SVM =====\n')

    print('===== BEGIN RESULTS FOR CNN =====\n')
    cnn.evaluate(test_images, test_labels, args.use_local_models)
    print('===== END RESULTS FOR CNN =====\n')

    print('===== BEGIN RESULTS FOR NN =====\n')
    nn.evaluate(test_images, test_labels, args.use_local_models)
    print('===== END RESULTS FOR NN =====\n')

    print('===== BEGIN RESULTS FOR TRANSFER NN =====\n')
    transferNN.evaluate(s_test_images, s_test_labels, args.use_local_models)
    print('===== END RESULTS FOR TRANSFER NN =====\n')

    print('\n==================== END RESULTS ====================\n')

  ############ Step 5 - Decide which model has performed best on our data ############ 

if __name__ == "__main__":
    main()