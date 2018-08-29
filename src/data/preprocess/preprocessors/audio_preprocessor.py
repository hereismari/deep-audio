'''
Author: Marianne Linhares, mariannelinharesm@gmail.com
Last Change: 08/2018
'''
import matplotlib.pyplot as plt
import librosa
import scipy

import numpy as np
import os
import glob


class AudioPreprocessor(object):
    @staticmethod
    def wav_to_spectrogram(wav, sr=22050, n_fft=2048):
        '''
            Preprocessing similar as presented in:
                Comparison of Time-Frequency Representations
                for Environmental Sound Classification using CNNs:
                https://arxiv.org/pdf/1706.07156.pdf, Muhammad Huzaifah
        '''
        # Generate spectrogram
        spectrogram = np.abs(librosa.stft(wav, sr=sr, n_fft=n_fft))
        # Convert to logarithmic scale 
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        return spectrogram


    @staticmethod
    def wav_to_melspectrogram(wav, sr=22050, n_fft=2048, n_mels=64):
        # Calculate melspectrogram
        spectrogram = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, n_mels=n_mels)
        # Convert to logarithmic scale
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram


    @staticmethod
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
        if not os.path.exists(os.path.dirname(output_path)): 
            os.makedirs(output_path)
        librosa.output.write_wav(output_path, x, sr)
        return x


    @staticmethod
    def wav_file_to(filepath, n_fft=2048, to='spectrogram', normalized=True, save_as=None):
        # Open wav file as mono
        wav, sr = librosa.load(filepath)

        if to == 'spectrogram':
            spectrogram = AudioPreprocessor.wav_to_spectrogram(wav, sr, n_fft)
        elif to == 'melspectrogram':
            spectrogram = AudioPreprocessor.wav_to_melspectrogram(wav, sr, n_fft, n_mels=64)
        else:
            raise Exception('format %s is unknown.' % to)
        
        if normalized:
            # Mean = 0, Std = 1
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
            assert spectrogram.mean() < 1e-4
            assert spectrogram.std() -1 < 1e-4

        if save_as:
            if not os.path.exists(os.path.dirname(output_path)): 
                os.makedirs(output_path)
            np.save(output_path, spectrogram)
        
        return spectrogram

