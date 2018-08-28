import tensorflow as tf
import numpy as np


class DataSource(object):
    def __init__(self, path, batch_size=16, buffer_size=100):
        self._path = path
        self._data = np.load(path)
        self._data = (self._data - np.mean(self._data)) / np.std(self._data)
        self.dataset = tf.data.Dataset.from_tensor_slices(self._data).map(lambda x: tf.reshape(x, (x.shape[0], x.shape[1], 1))).shuffle(buffer_size).batch(batch_size)