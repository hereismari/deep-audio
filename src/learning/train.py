import tensorflow as tf
tf.enable_eager_execution()

import numpy as np

from data_sources.data_source import DataSource
from optimizers.optimizer import Optimizer
import models.model_factory as mf
import utilities.utils as utils


def train():
    ds_train = DataSource('audio_files/ae_dataset/train_data.npy', 'audio_files/ae_dataset/train_labels.npy', classes_dict='audio_files/ae_dataset/classes')
    ds_eval  = DataSource('audio_files/ae_dataset/eval_data.npy', 'audio_files/ae_dataset/eval_labels.npy', classes_dict='audio_files/ae_dataset/classes')
    ds_test  = DataSource('audio_files/ae_dataset/test_data.npy', 'audio_files/ae_dataset/test_labels.npy', classes_dict='audio_files/ae_dataset/classes')
   
    import ipdb; ipdb.set_trace()

    model = mf.build_model('CNN', input_shape=ds_train._data[0].shape, num_classes = ds_train.num_classes)
    optimizer = Optimizer('Adam', 'RMSE')

    EPOCHS = 10
    for epoch in range(EPOCHS):
        train_losses = []
        train_accuracy = 0
        train_instances = 0
        for (batch, (img_tensor, label)) in enumerate(ds_train.dataset):
            train_loss, predicted = optimizer.compute_and_apply_gradients(model, img_tensor, label)
            train_losses.append(train_loss)
            train_accuracy += sum(np.argmax(predicted, axis=1) == label.numpy())
            train_instances += label.numpy().shape[0]
            print ('Epoch {} Batch {} Train Loss {:.6f}'.format(epoch + 1, batch + 1, sum(train_losses)/len(train_losses)))


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


if __name__ == '__main__':
    train()