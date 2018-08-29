import numpy as np
from sklearn.model_selection import train_test_split


class PartitionPreprocessor(object):
    @staticmethod
    def dataset_partition(data, labels=None, test_size=0.25):
        if labels is None:
            return train_test_split(data, test_size=0.25)
        else:
            return train_test_split(data, labels, test_size=0.25)
