import tensorflow as tf
tf.enable_eager_execution()

import tensorflowjs as tfjs

import numpy as np

from data_sources.data_source import DataSource
from optimizers.optimizer import Optimizer
import models.model_factory as mf
import utilities.utils as utils

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def train():
    ds_train = DataSource('audio_files/speech/train_data.npy', 'audio_files/speech/train_labels.npy', classes_dict='audio_files/speech/classes')
    ds_eval  = DataSource('audio_files/speech/eval_data.npy', 'audio_files/speech/eval_labels.npy', classes_dict='audio_files/speech/classes')
    ds_test  = DataSource('audio_files/speech/test_data.npy', 'audio_files/speech/test_labels.npy', classes_dict='audio_files/speech/classes')

    model = mf.build_model('CNN', input_shape=ds_train.input_shape, num_classes = ds_train.num_classes)
    optimizer = Optimizer('Adam')

    EPOCHS = 1
    for epoch in range(EPOCHS):
        train_losses = []
        train_accuracy = 0
        train_instances = 0
        for (batch, (img_tensor, label)) in enumerate(ds_train.dataset):
            train_loss, predicted = optimizer.compute_and_apply_gradients(model, img_tensor, label)
            train_losses.append(train_loss)
            train_accuracy += sum(np.argmax(predicted, axis=1) == label.numpy())
            train_instances += label.numpy().shape[0]
            # print ('Epoch {} Batch {} Train Loss {:.6f}'.format(epoch + 1, batch + 1, sum(train_losses)/len(train_losses)))


        eval_losses = []
        accuracy = 0
        instances = 0
        for (batch, (img_tensor, label)) in enumerate(ds_eval.dataset):
            eval_loss, predicted = model.compute_loss(img_tensor, label, training=False)
            accuracy += sum(np.argmax(predicted, axis=1) == label.numpy())
            instances += label.numpy().shape[0]
            eval_losses.append(eval_loss)

        print ('Epoch {} Train Accuracy: {:.6f} | Test Accuracy: {:.6f}'.format(epoch +1, train_accuracy/train_instances, accuracy/instances))
        print ('Epoch {} Train Loss {:.6f} | Eval Loss {:.6f}'.format(epoch + 1, sum(train_losses)/len(train_losses), sum(eval_losses)/len(eval_losses)))


    eval_losses = []
    accuracy = 0
    instances = 0
    for (batch, (img_tensor, label)) in enumerate(ds_test.dataset):
        eval_loss, predicted = model.compute_loss(img_tensor, label, training=False)
        accuracy += sum(np.argmax(predicted, axis=1) == label.numpy())
        instances += label.numpy().shape[0]
        eval_losses.append(eval_loss)
    
    print ('Epoch {} Train Accuracy: {:.6f} | Test Accuracy: {:.6f}'.format(epoch +1, train_accuracy/train_instances, accuracy/instances))
    print ('Epoch {} Train Loss {:.6f} | Eval Loss {:.6f}'.format(epoch + 1, sum(train_losses)/len(train_losses), sum(eval_losses)/len(eval_losses)))


    import ipdb; ipdb.set_trace()

    saver = tf.train.Checkpoint(model=model, optimizer=optimizer.optimizer)
    saver.save('/home/marianne/deep-audio/saver/save')

    tfjs.converters.convert_tf_saved_model('/home/marianne/deep-audio/saver/save.ckpt', output_node_names=['test'], output_dir='/home/marianne/deep-audio/')


if __name__ == '__main__':
    train()