import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from utils.print import pretty_print_confusion

# Transfer Neural Network from our CNN
class TransferNN:
  def __init__(self, base_model, model_filename = 'transfer.h5', epochs = 5, load_base=False):
    self.model_filename = model_filename
    self.epochs = epochs
    base_model = base_model.model

    if load_base == True:
      base_model = tf.keras.models.load_model(f'main/trained_models/base_{self.model_filename}')

    self.model = tf.keras.models.Sequential([
      base_model,
      tf.keras.layers.Dense(20),
      tf.keras.layers.Dense(65),
      tf.keras.layers.Dense(20)
    ])

    # Define a loss function for our model
    self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  def train(self, train_images, train_labels, verbose = True):
    # Load the data
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
