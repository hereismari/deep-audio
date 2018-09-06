import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization
from models.model import Model

class DAE_CNN(Model):
    def __init__(self, input_shape):
        super().__init__()

        # BatchNorm
        self.batchNorm = BatchNormalization()

        # Encoding
        self.conv1 = Conv2D(filters=32, kernel_size=3,
                                            input_shape=input_shape, activation='relu',
                                            padding='same', strides=(5, 5))
        self.conv2 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', strides=(5, 2))
        self.flat = Flatten()
        self.encode = Dense(10)

        # Decoding
        self.dense2 = Dense(units=41*43*32, activation=tf.nn.relu)
        self.reshape = Reshape((41, 43, 32))

        self.conv3 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', strides=(5, 2))
        self.conv4 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', strides=(5, 5))

        self.conv5 = Conv2DTranspose(filters=1, kernel_size=3, padding='same', strides=(1, 1))

    def call(self, x):
        # Noise
        original_x = x
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=0.1, dtype=tf.float32)
        x = noise + x
        # Encoder
        x = self.batchNorm(self.conv1(x))
        x = self.batchNorm(self.conv2(x))
        encode = self.encode(self.flat(x))

        # Decoder
        x = self.reshape(self.dense2(encode))
        x = self.batchNorm(self.conv3(x))

        x = self.batchNorm(self.conv4(x))

        decode = self.conv5(x)
        return encode, decode
    
    def compute_loss(self, x):
        encode, decode = self.call(x)
        return tf.reduce_mean(tf.pow(decode - x, 2))