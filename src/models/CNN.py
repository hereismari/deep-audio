import tensorflow as tf
from models.model import Model

class CNN(Model):
  def __init__(self, image_shape):
    super().__init__()

    self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=(1025, 430, 1))
    self.pool1 = tf.keras.layers.MaxPool2D()
    self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3)
    self.pool2 = tf.keras.layers.MaxPool2D()
    self.flat = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(128, activation='relu')
    self.drop = tf.keras.layers.Dropout(rate=0.25)
    self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')


  def call(self, x):
    x = self.pool1(self.conv1(x))
    x = self.pool2(self.conv2(x))
    x = self.pool2(self.conv2(x))
    x = self.flat(x)
    x = self.fc1(x)
    x = self.drop(x)
    x = self.fc2(x)
    return x