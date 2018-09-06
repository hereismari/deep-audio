from data.download.utilities.utils import *
import numpy as np
import librosa
import os

def spectrogram_to_wav(spectrogram, output_path, sr=22050, n_fft=2048):
    '''Code from: https://github.com/DmitryUlyanov/neural-style-audio-tf/blob/master/neural-style-audio-tf.ipynb'''
    # Spectrogram to Wav
    a = np.zeros_like(spectrogram)
    a[:spectrogram.shape[0],:] = np.exp(spectrogram) - 1

    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        S = a * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft))
    
    # Save wav
    if os.path.dirname(output_path) != '' and not os.path.exists(os.path.dirname(output_path)): 
        os.makedirs(output_path)
    librosa.output.write_wav(output_path, x, sr)
    return x