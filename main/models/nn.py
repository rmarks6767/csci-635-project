import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from utils.print import pretty_print_confusion

# Neural Network
class NN:
  def __init__(self, model_filename = 'nn.h5', epochs = 5):
    self.model_filename = model_filename
    self.epochs = epochs

    # Define two hidden layers for the model
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(784),
      tf.keras.layers.Dense(100),
      tf.keras.layers.Dense(64),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(20, activation="softmax")
    ])

    # Define a loss function for our model
    self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

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
    self.model.evaluate(test_images, test_labels, verbose=2)
    print(classification_report(test_labels, predictions))
