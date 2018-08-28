import tensorflow as tf
tf.enable_eager_execution()

from data_sources.data_source import DataSource
from optimizers.optimizer import Optimizer
import models.model_factory as mf
import utilities.utils as utils


def train():
    ds = DataSource('audio_files/preprocessed.npz.npy')
    model = mf.build_model('DAE_CNN', input_shape=ds._data[0].shape)
    optimizer = Optimizer('Adam', 'RMSE')

    EPOCHS = 100
    for epoch in range(EPOCHS):
        losses = []
        for (batch, (img_tensor)) in enumerate(ds.dataset):
            loss = optimizer.compute_and_apply_gradients(model, img_tensor)
            losses.append(loss)

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, sum(losses)/len(losses)))
        #if (epoch % 100 == 0 and epoch > 0) or epoch == 299:
        #    encode, decode = model(img_tensor)
        #    utils.spectrogram_to_wav(decode[0].numpy().reshape(1025, 430), 'test_epoch_%d.wav' % epoch)
        #    utils.spectrogram_to_wav(img_tensor[0].numpy().reshape(1025, 430), 'orig_epoch_%d.wav' % epoch)

    for (batch, (img_tensor)) in enumerate(ds.dataset):
        encode,decode = model(img_tensor)
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    train()