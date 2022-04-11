from tabnanny import verbose
import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import load_all_data, load_data
import os
from skimage.io import imread
import numpy as np

def run_training():
  base_model = tf.keras.models.load_model('cnn_english_10.h5')

  (train_images, train_labels), (test_images, test_labels) = load_data('sylheti')
  train_images, test_images = train_images / 255.0, test_images / 255.0

  model = tf.keras.Sequential([
    base_model,
    # tf.keras.layers.RandomZoom(.5, .2),
    tf.keras.layers.Dense(10)
  ])

  predictions = model(train_images[:1]).numpy()

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  loss_fn(train_labels[:1], predictions).numpy()

  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  model.fit(train_images, train_labels, epochs=10)
  model.evaluate(test_images, test_labels, verbose=2)
  # model.save('cnn_transfer_eng_10.h5')

def test_model():
  model = tf.keras.models.load_model('cnn_transfer_eng_10.h5')

  images = []
  for i in range(0, 10):
    images.append(imread(os.path.join(os.path.dirname(__file__), os.path.join(f'{i}-test.png')), as_gray=True))

  image = np.array(images)

  prediction = model.predict(image)

  for i in range(0, 10):
    print(f'{np.argmax(prediction[i])} should be {i}')

run_training()
# test_model()