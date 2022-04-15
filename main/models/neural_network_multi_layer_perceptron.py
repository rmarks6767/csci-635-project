import os.path
from sklearn.neural_network import MLPClassifier
from mnist import MNIST
from joblib import dump, load
from data_loader import load_all_data
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from main.data.data_loader import load_all_data


class MultiLayerPerceptron:
    def __init__(self, model_filename = 'MLP.joblib'):
        self.model = MLPClassifier(hidden_layer_sizes=(588, 392, 196),
                                   activation='relu',
                                   solver='sgd',
                                   alpha=0.0001,
                                   batch_size='auto',
                                   learning_rate='constant',
                                   learning_rate_init=0.001,
                                   random_state=1)
        self.model_filename = model_filename

    def train(self):
        # Get the images from the directory
        (train_images, train_labels), (test_images, test_labels) = load_all_data(True)
        self.model.fit(train_images, train_labels)
        dump(self.model, self.model_filename)

        # Run some predictions and see how accurate they are with a confusion matrix
        predictions = self.model.predict(test_images)

        # Output the confusion matrix and model report
        self.print_results(test_labels, predictions)

    def test(self, image, correct_output):
        # Load the model from the file we saved it to
        model = load(self.model_filename)

        # Structure the image how the model will read it
        image = np.array([image])

        # Predict the image
        prediction = model.predict(image)
        print(f'{np.argmax(prediction[0])} should be {correct_output}')

    def print_results(self, test_labels, predictions):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print("\nConfusion matrix:")
        print("Labels: {0}\n".format(",".join(labels)))
        print(confusion_matrix(test_labels, predictions, labels=labels))
        print("\nClassification report:")
        print(classification_report(test_labels, predictions))


def main1():
    # PLAYING AROUND With ENGLISH DATA
    (train_data, train_label), (test_data, test_label) = load_all_data(True)
    # Insight into choosing hidden layers and their nodes:
    # https://www.linkedin.com/pulse/choosing-number-hidden-layers-neurons-neural-networks-sachdev
    # 3 hidden layers
    clf = MLPClassifier(hidden_layer_sizes=(588, 392, 196),
                        activation='relu',
                        solver='sgd',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        random_state=1)
    clf.fit(train_data, train_label)
    prediction = clf.predict(test_data)
    idx = 0
    correct = 0
    total = len(english_test_label)
    while idx < total:
        if prediction[idx] == test_label[idx]:
            correct += 1
        idx += 1
    print(correct/total)


def main():
    mlp = MultiLayerPerceptron('m1')
    mlp.train()


if __name__ == "__main__":
    main()
