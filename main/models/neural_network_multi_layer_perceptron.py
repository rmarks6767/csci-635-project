import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from main.utils.print import pretty_print_confusion

from main.data.data_loader import load_all_data


class MultiLayerPerceptron:
    def __init__(self, model_filename='MLP.h5', epochs = 5):

        self.epochs = epochs
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=784),
            tf.keras.layers.Dense(588, activation="sigmoid"),
            tf.keras.layers.Dense(392, activation="sigmoid"),
            tf.keras.layers.Dense(196, activation="sigmoid"),
            tf.keras.layers.Dense(20, activation="softmax")
        ])

        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

        self.model_filename = model_filename


    def train(self, train_images, train_labels, verbose = True):
        train_images = train_images / 255.0

        # Define predictions and loss function outputs
        predictions = self.model(train_images[:1]).numpy()
        self.loss_func(train_labels[:1], predictions).numpy()

        # Compile and run the model on our data
        self.model.compile(optimizer='adam', loss=self.loss_func, metrics=['accuracy'])
        self.model.fit(train_images, train_labels, epochs=self.epochs, verbose=verbose)

        # Save the model so we can use it later
        self.model.save(f'main/trained_models/{self.model_filename}')


    def evaluate(self, test_images, test_labels, load = False):
        test_images = test_images / 255.0

        # If we want to load the load we can use the load tag
        if load:
            self.model = tf.keras.models.load_model(f'main/trained_models/{self.model_filename}')

        # Run predictions on our test data and evaluate results
        predictions = []
        for p in self.model.predict(test_images):
            predictions.append(np.argmax(p))

        # Create a confusion matrix for our data
        print('\nConfusion Matrix:')
        pretty_print_confusion(tf.math.confusion_matrix(test_labels, predictions))

        # Print the classification report
        print('\nClassification report:')
        report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
        print(classification_report(test_labels, predictions, zero_division=0))

        return report


def main():
    mlp = MultiLayerPerceptron(model_filename='m1.h5')
    (train_images, train_labels), (test_images, test_labels) = load_all_data(True)
    #mlp.train(train_images, train_labels)
    mlp.evaluate(test_images, test_labels, load=True)


if __name__ == "__main__":
    main()
