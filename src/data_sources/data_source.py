import tensorflow as tf
import numpy as np
import pickle


class DataSource(object):
    def __init__(self, data_path, data_labels, batch_size=64, buffer_size=None, classes_dict=None, num_classes=10):
        self._data_path = data_path
        self._data_labels = data_labels
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._classed_dict = classes_dict

        self._data = np.load(data_path)
        self._labels = np.load(data_labels)

        self.input_shape = self._data[0].shape

        if self._classed_dict is not None:
            self._classes = pickle.load(open(classes_dict, 'rb'))
            self.num_classes = len(self._classes)
        else:
            self._classes = {'unknown': 0}
            self.num_classes = num_classes
    
        print(self._classes)

        if self._buffer_size is None:
            self._buffer_size = len(self._data)

        self._data_dataset = tf.data.Dataset.from_tensor_slices(self._data)
        self._data_dataset = self._data_dataset.map(lambda x: tf.reshape(x, (x.shape[0], x.shape[1], 1)))
        self._data_dataset = self._data_dataset.map(lambda x: tf.cast(x, tf.float32))

        self._labels_dataset = tf.data.Dataset.from_tensor_slices(self._labels)
        self._labels_dataset = self._labels_dataset.map(lambda x: tf.one_hot(x, self.num_classes))

        self.dataset = tf.data.Dataset.zip((self._data_dataset, self._labels_dataset))
        self.dataset = self.dataset.shuffle(self._buffer_size).batch(batch_size)
    
    def __len__(self):
        return len(self._labels)