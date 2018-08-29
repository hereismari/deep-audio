import tensorflow as tf
import numpy as np


class DataSource(object):
    def __init__(self, data_path, data_labels, batch_size=128, buffer_size=100):
        self._data_path = data_path
        self._data_labels = data_labels
        
        self._data = np.load(path)
        self._labels = np.load(labels)

        self.dataset = tf.data.Dataset.from_tensor_slices(self._data, self._labels).map(lambda x, y: tf.reshape(x, (x.shape[0], x.shape[1], 1)), y)
        self.dataset = self.dataset.shuffle(buffer_size).batch(batch_size)