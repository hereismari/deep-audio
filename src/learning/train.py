import tensorflow as tf
tf.enable_eager_execution()

from data_sources.data_source import DataSource
from optimizers.optimizer import Optimizer
import models.model_factory as mf



def train():
    ds = DataSource('audio_files/preprocessed.npz.npy')
    model = mf.build_model('DAE_CNN', input_shape=ds._data[0].shape)
    optimizer = Optimizer('Adam', 'RMSE')

    EPOCHS = 100
    for epoch in range(EPOCHS):
        for (batch, (img_tensor)) in enumerate(ds.dataset):
            loss = optimizer.compute_and_apply_gradients(model, img_tensor)
    
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, loss))
if __name__ == '__main__':
    train()