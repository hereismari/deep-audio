import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Softmax, Flatten, Dense, Dropout, BatchNormalization, Activation
from models.model import Model


class CNN(Model):
  def __init__(self, input_shape, num_classes=2):
    super().__init__()

    self.batchnorm1 = BatchNormalization()
    self.batchnorm2 = BatchNormalization()


    self.conv1 = Conv2D(filters=128, kernel_size=3, input_shape=input_shape, activation='relu', padding='same')
    self.conv2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')
    self.pool1 = MaxPool2D()

    self.conv3 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')
    self.conv4 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')
    self.pool2 = MaxPool2D()

    self.flat = Flatten()
    self.dense1 = Dense(256, activation='relu')
    self.dropout = Dropout(0.3)
    self.dense2 = Dense(128, activation='relu')
    self.final = Dense(num_classes)


  def call(self, x, training=True):
    x = self.conv1(x)
    x = self.pool1(self.conv2(x))
    
    x = self.conv3(x)
    x = self.pool2(self.conv4(x))
    
    x = self.flat(x)
    x = self.dense1(x)
    if training:
      x = self.dropout(x)
    
    x = self.dense2(x)
    x = self.final(x)
    return x
  

  def compute_loss(self, x, y, training=True):
    prediction = self.call(x, training=training)
    return tf.losses.sparse_softmax_cross_entropy(labels=y,  logits=prediction), prediction