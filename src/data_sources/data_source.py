import tensorflow as tf
import numpy as np


class DataSource(object):
    def __init__(self, path):
        self._path = path
        self._data = np.load(path)
        self._data = self._data / np.max(self._data)
        import ipdb; ipdb.set_trace()
        self.dataset = tf.data.Dataset.from_tensor_slices(self._data).map(lambda x: tf.reshape(x, (x.shape[0], x.shape[1], 1))).batch(16)