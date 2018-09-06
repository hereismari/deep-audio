import tensorflow as tf
tf.enable_eager_execution()

import tensorflowjs as tfjs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Softmax, Flatten, Dense, Dropout, BatchNormalization, Activation


import numpy as np

from data_sources.data_source import DataSource
from optimizers.optimizer import Optimizer
import models.model_factory as mf
import utilities.utils as utils

from data.preprocess.preprocessors.audio_preprocessor import AudioPreprocessor


def train():
    PATH = 'audio_files/transfer/'
    ds_train = DataSource(PATH + 'train_data.npy', PATH + 'train_labels.npy', classes_dict=PATH + 'classes')
    ds_eval  = DataSource(PATH + 'eval_data.npy', PATH + 'eval_labels.npy', classes_dict=PATH + 'classes')
    ds_test  = DataSource(PATH + 'test_data.npy', PATH + 'test_labels.npy', classes_dict=PATH + 'classes')


    model = tf.keras.Sequential([
            Conv2D(filters=128, kernel_size=3, input_shape=(64, 35, 1), activation='relu', padding='same'),
            Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPool2D(),
            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPool2D(),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(4, activation='softmax')
        ])
    
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('model.h5')
    for layer in model.layers:
        layer.trainable = False

    x = Dense(128)(model.layers[-3].output)
    x = Dense(3, activation='softmax')(x)
    model2 = Model(inputs=model.input, outputs=[x])
    model2.summary()
    model2.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

    model2.fit(ds_train._data.reshape(-1, 64, 35, 1), tf.keras.utils.to_categorical(ds_train._labels),
              epochs=20,
              validation_data=(ds_eval._data.reshape(-1, 64, 35, 1), tf.keras.utils.to_categorical(ds_eval._labels)),
              verbose=2,
              batch_size=64)

    print(model2.evaluate(x=ds_test._data.reshape(-1, 64, 35, 1), y=tf.keras.utils.to_categorical(ds_test._labels)))
    model2.save('model2.h5')
    import ipdb; ipdb.set_trace()

    x = AudioPreprocessor.wav_file_to('dog.wav', to='melspectrogram')
    y = model2.predict(x.reshape(-1, 64, 35, 1))
    print(y)

if __name__ == '__main__':
    train()