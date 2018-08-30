import tensorflow as tf


class Optimizer(object):

    _OPTIMIZERS = {
        'Adam': tf.train.AdamOptimizer
    }

    def __init__(self, optimizer_name, loss_function_name, learning_rate=0.0005, **kwargs):
        if optimizer_name in self._OPTIMIZERS:
            self.optimizer_name = optimizer_name
            self.optimizer = self._OPTIMIZERS[optimizer_name](learning_rate=learning_rate, **kwargs)
        else:
            raise Exception('Optimizer unknown %s' % optimizer_name)


    def compute_gradients(self, model, x, y=None):
        with tf.GradientTape() as tape:
            loss, predicted = model.compute_loss(x, y)
        return tape.gradient(loss, model.trainable_variables), loss, predicted


    def apply_gradients(self, gradients, variables, global_step=None):
        self.optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


    def compute_and_apply_gradients(self, model, x, y=None, global_step=None):
        gradients, loss, predicted = self.compute_gradients(model, x, y)
        self.apply_gradients(gradients, model.trainable_variables, global_step=global_step)
        return loss, predicted