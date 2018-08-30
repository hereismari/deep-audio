import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Softmax, Flatten, Dense, Dropout, BatchNormalization
from models.model import Model


class CNN(Model):
  def __init__(self, input_shape, num_classes=2):
    super().__init__()

    self.batchnorm = BatchNormalization()
    self.conv1 = Conv2D(filters=64, kernel_size=3, input_shape=input_shape, activation='relu')
    self.pool1 = MaxPool2D()
    
    self.conv2 = Conv2D(filters=64, kernel_size=3, activation='relu')
    self.pool2 = MaxPool2D(pool_size=(2, 2))

    self.conv3 = Conv2D(filters=128, kernel_size=3, activation='relu')
    
    self.conv4 = Conv2D(filters=128, kernel_size=3, activation='relu')
    self.pool4 = MaxPool2D()

    self.conv5 = Conv2D(filters=128, kernel_size=3, activation='relu')
    self.conv6 = Conv2D(filters=128, kernel_size=1, activation='relu')
    self.conv7 = Conv2D(filters=128, kernel_size=1, activation='relu')
    self.pool3 = AvgPool2D()

    self.flat = Flatten()
    self.dense1 = Dense(1024, activation='relu')
    self.dropout = Dropout(0.5)
    self.dense2 = Dense(1024)
    self.final = Dense(num_classes)


  def call(self, x, training=True):
    #x = self.batchnorm(self.conv1(x), training=training)
    x = self.conv1(x)
    x = self.pool2(self.conv2(x))

    # x = self.batchnorm(self.conv3(x), training=training)
    x = self.conv3(x)
    x = self.pool1(self.conv4(x))

    x = self.conv3(x)
    x = self.pool1(self.conv4(x))

    # x = self.batchnorm(self.conv5(x), training=training)
    # x = self.conv5(x)

    x = self.flat(x)
    x = self.dense1(x)
    if training:
      x = self.dropout(x)
    
    # x = self.dense2(x)
    x = self.final(x)
    return x
  

  def compute_loss(self, x, y, training=True):
    prediction = self.call(x, training=training)
    return tf.losses.sparse_softmax_cross_entropy(labels=y,  logits=prediction), prediction