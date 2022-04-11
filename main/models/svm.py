import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import load_all_data
import numpy as np

# Support Vector Machine
class SVM:
  def __init__(self, model_name = 'cnn.h5'):
    self.model_name = model_name

    # Model will have 3 convolution layers into three hidden NN layers,
    # using Softmax for activation on the final layer
    self.model = models.Sequential([
      layers.Conv2D(28, (1,1), padding='same', activation="relu",input_shape=(28, 28, 1)),
      layers.MaxPooling2D((2, 2), strides=2),
      layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      layers.MaxPooling2D((2, 2), strides=2),
      layers.Conv2D(64, (3,3), padding='same', activation="relu"),
      layers.MaxPooling2D((2, 2), strides=2),
      layers.Flatten(),
      layers.Dense(100, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dropout(0.2),
      layers.Dense(10, activation="softmax")
    ])

    # Define a loss function for our model
    self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

  def train(self):
    # Load the data
    (train_images, train_labels), (test_images, test_labels) = load_all_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Define predictions and loss function outputs
    predictions = self.model(train_images[:1]).numpy()
    self.loss_func(train_labels[:1], predictions).numpy()

    # Compile and run the model on our data
    self.model.compile(optimizer='adam', loss=self.loss_func, metrics=['accuracy'])
    self.model.fit(train_images, train_labels, epochs=self.epochs)
    self.model.evaluate(test_images, test_labels, verbose=2)

    # Save the model so we can use it later
    self.model.save(self.model_name)

  def test(self, image, correct_output):
    # Load the model from the file we saved it to
    model = tf.keras.models.load_model(self.model_name)

    # Structure the image how the model will read it
    images = []
    image = np.array(images)

    # Predict the image
    prediction = model.predict(image)
    print(f'{np.argmax(prediction[0])} should be {correct_output}')
