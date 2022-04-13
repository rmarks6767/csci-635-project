import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from sklearn.metrics import classification_report
from utils.print import pretty_print_confusion

# Transfer Neural Network from MobileNet_v2
class TransferNN:
  def __init__(self, model_filename = 'transfer_resnet.h5', epochs = 5):
    self.model_filename = model_filename
    self.epochs = epochs

    resnet_v1 ="https://tfhub.dev/google/supcon/resnet_v1_200/imagenet/classification/1"

    self.model = tf.keras.models.Sequential([
      hub.KerasLayer(
        resnet_v1,
        input_shape=(224, 224, 3),
        trainable=False
      ),
      tf.keras.layers.Dense(20)
    ])

    # Define a loss function for our model
    self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  def train(self, train_images, train_labels):
    # Load the data
    train_images = train_images / 255.0

    # Resize images to work with resnet
    resize_train = []
    for img in train_images:
      resize_train.append(
        tf.image.grayscale_to_rgb(
          tf.expand_dims(cv2.resize(img, (224, 224)), -1)
        )
      )
    
    resize_train = np.array(resize_train)

    # Define predictions and loss function outputs
    predictions = self.model(resize_train[:1]).numpy()
    self.loss_func(train_labels[:1], predictions).numpy()

    # Compile and run the model on our data
    self.model.compile(optimizer='adam', loss=self.loss_func, metrics=['accuracy'])
    self.model.fit(resize_train, train_labels, epochs=self.epochs)

    # Save the model so we can use it later
    self.model.save(self.model_filename)

  def evaluate(self, test_images, test_labels, load = False):
    test_images = test_images / 255.0

    # We have to resize the test images to work with the other size
    # (224, 224, 3)
    resize_test = []
    for img in test_images:
      resize_test.append(
        tf.image.grayscale_to_rgb(
          tf.expand_dims(cv2.resize(img, (224, 224)), -1)
        )
      )

    resize_test = np.array(resize_test)

    # If we want to load the load we can use the load tag
    if load:
      self.model = tf.keras.models.load_model(self.model_filename)

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
