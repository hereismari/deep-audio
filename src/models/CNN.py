import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Softmax
from models.model import Model

class CNN(Model):
  def __init__(self, input_shape):
    super().__init__()

    self.conv1 = Conv2D(filters=64, kernel_size=3, input_shape=input_shape, activation='relu')
    self.conv2 = Conv2D(filters=64, kernel_size=3, activation='relu')
    self.pool1 = MaxPool2D()

    self.conv3 = Conv2D(filters=128, kernel_size=3, activation='relu')
    self.conv4 = Conv2D(filters=128, kernel_size=3, activation='relu')
    self.pool2 = MaxPool2D()

    self.conv5 = Conv2D(filters=128, kernel_size=3, activation='relu')
    self.conv6 = Conv2D(filters=128, kernel_size=1, activation='relu')
    self.conv7 = Conv2D(filters=128, kernel_size=1, activation='relu')
    self.pool3 = AvgPool2D()
    self.classification = Softmax()


  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(self.conv2(x))

    x = self.conv3(x)
    x = self.pool2(self.conv4(x))

    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.pool3(x)
    x = self.classification(x)
    return x