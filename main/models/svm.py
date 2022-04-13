import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from utils.print import pretty_print_confusion

class SVM:
  def __init__(self, model_filename = 'svm.joblib'):
    # Best parameters that were found after running on a smaller data set
    params = [{ "kernel": ["linear"], "C": [1] }]

    # Create a probability based SVM
    svm = SVC(probability=True)
    self.model = GridSearchCV(svm, params, cv=2, n_jobs=6, verbose=3)
    self.model_filename = model_filename

  def train(self, train_images, train_labels):
    # For the SVM, we have to flatten our data
    train_images = train_images.flatten().reshape(len(train_images), 784)

    # Use Grid Search to fit an Support Vector Machine
    self.model.fit(train_images, train_labels)

    # Save the model that we just created
    dump(self.model, self.model_filename) 

  def evaluate(self, test_images, test_labels, should_load = False):
    # If we want to load the load we can use the load tag
    if should_load:
      self.model = load(self.model_filename)

    # Run predictions on our test data and evaluate results
    predictions = []
    for p in self.model.predict(test_images):
      predictions.append(np.argmax(p))

    # Create a confusion matrix for our data
    print('\nConfusion Matrix:')
    pretty_print_confusion(tf.math.confusion_matrix(test_labels, predictions))

    # Print the classification report
    print('\nClassification report:')
    self.model.evaluate(test_images, test_labels, verbose=2)
    print(classification_report(test_labels, predictions))
