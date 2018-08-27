import tensorflow as tf
from models.model import Model

class DAE_CNN(Model):
    def __init__(self, input_shape):
        super().__init__()

        # Encoding
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                            input_shape=input_shape, activation='relu',
                                            padding='same', strides=(5, 5))
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', strides=(5, 2))
        self.flat = tf.keras.layers.Flatten()
        self.encode = tf.keras.layers.Dense(10)

        # Decoding
        self.dense2 = tf.keras.layers.Dense(units=41*43*32, activation=tf.nn.relu)
        self.reshape = tf.keras.layers.Reshape((41, 43, 32))

        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', strides=(5, 2))
        self.conv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', strides=(5, 5))

        self.conv5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, padding='same', strides=(1, 1))

    def call(self, x):
        # Noise
        original_x = x
        # x = tf.random_normal(shape=x.shape) + x
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        encode = self.encode(self.flat(x))

        # Decoder
        x = self.reshape(self.dense2(encode))
        x = self.conv3(x)

        x = self.conv4(x)

        decode = self.conv5(x)
        return encode, decode
    
    def compute_loss(self, x):
        encode, decode = self.call(x)
        return tf.reduce_sum(tf.pow(tf.subtract(decode, x), 2))