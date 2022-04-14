import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from utils.print import pretty_print_confusion

# Convolutional Neural Network
class CNN:
  def __init__(self, model_filename = 'cnn.h5', epochs = 5):
    self.model_filename = model_filename
    self.epochs = epochs

    # Model will have 3 convolution layers into three hidden NN layers,
    # using Softmax for activation on the final layer
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(28, (1,1), padding='same', activation="relu",input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),
      tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),
      tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      tf.keras.layers.MaxPooling2D((2, 2), strides=2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(100, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
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
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    print(classification_report(test_labels, predictions, zero_division=0))

    return report
