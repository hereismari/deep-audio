import tensorflow as tf


class Model(tf.keras.Model):

    def __init__(self):
        super().__init__()
    
    def gru(self, units):
        # If you have a GPU is recommended using the CuDNNGRU layer (it provides a 
        # significant speedup).
        if tf.test.is_gpu_available():
            return tf.keras.layers.CuDNNGRU(units, 
                                            return_sequences=True, 
                                            return_state=True, 
                                            recurrent_initializer='glorot_uniform')
        else:
            return tf.keras.layers.GRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_activation='sigmoid', 
                                    recurrent_initializer='glorot_uniform')