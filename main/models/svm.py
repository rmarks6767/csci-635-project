import tensorflow as tf
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
    self.model = GridSearchCV(svm, params, cv=2, n_jobs=6, verbose=False)
    self.model_filename = model_filename

  def train(self, train_images, train_labels):
    # For the SVM, we have to flatten our data and scale (for performance)
    train_images = train_images.flatten().reshape(len(train_images), 784)
    # Use Grid Search to fit an Support Vector Machine
    self.model.fit(train_images, train_labels)

    # Save the model that we just created
    dump(self.model, f'main/trained_models/{self.model_filename}') 

  def evaluate(self, test_images, test_labels, should_load = False):
    # If we want to load the load we can use the load tag
    if should_load:
      self.model = load(f'main/trained_models/{self.model_filename}')

    # Have to flatten the test images to be run through the model
    test_images = test_images.flatten().reshape(len(test_images), 784)

    # Run predictions on our test data and evaluate results
    predictions = self.model.predict(test_images)

    # Create a confusion matrix for our data
    print('\nConfusion Matrix:')
    pretty_print_confusion(tf.math.confusion_matrix(test_labels, predictions))

    # Print the classification report
    print('\nClassification report:')
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    print(classification_report(test_labels, predictions, zero_division=0))

    return report
