from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from data_loader import load_all_data
from joblib import dump, load
import numpy as np

class SVM:
  def __init__(self, model_filename = 'svm.joblib'):
    # Best parameters that were found after running on a smaller data set
    params = [{ "kernel": ["linear"], "C": [1] }]

    # Create a probability based SVM
    svm = SVC(probability=True)
    self.model = GridSearchCV(svm, params, cv=2, n_jobs=6, verbose=3)
    self.model_filename = model_filename

  def train(self):
    # Get the images from the directory
    (train_images, train_labels), (test_images, test_labels) = load_all_data(True)

    # Use Grid Search to fit an Support Vector Machine
    self.model.fit(train_images, train_labels)

    # Save the model that we just created
    dump(self.model, self.model_filename) 

    # Run some predictions and see how accurate they are with a confusion matrix
    predictions = self.model.predict(test_images)

    # Output the confusion matrix and model report
    self.print_results(test_labels, predictions)

  def print_results(self, test_labels, predictions):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(test_labels, predictions, labels=labels))
    print("\nClassification report:")
    print(classification_report(test_labels, predictions))

  def test(self, image, correct_output):
    # Load the model from the file we saved it to
    model = load(self.model_filename) 

    # Structure the image how the model will read it
    image = np.array([image])

    # Predict the image
    prediction = model.predict(image)
    print(f'{np.argmax(prediction[0])} should be {correct_output}')