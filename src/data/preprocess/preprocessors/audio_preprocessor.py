'''
Author: Marianne Linhares, mariannelinharesm@gmail.com
Last Change: 08/2018
'''
import matplotlib.pyplot as plt
import librosa
import librosa.display
        
import pydub
import scipy

import numpy as np
import os
import glob


class AudioPreprocessor(object):
    @staticmethod
    def wav_to_spectrogram(wav, n_fft=2048):
        '''
            Preprocessing similar as presented in:
                Comparison of Time-Frequency Representations
                for Environmental Sound Classification using CNNs:
                https://arxiv.org/pdf/1706.07156.pdf, Muhammad Huzaifah
        '''
        # Generate spectrogram
        spectrogram = np.abs(librosa.stft(wav, n_fft=n_fft))
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
        librosa.output.write_wav(output_path, x, sr)
        return x


    @staticmethod
    def wav_file_to(filepath, n_fft=2048, to='spectrogram', normalized=True, save_as=None):
        # Open wav file as mono
        wav, sr = librosa.load(filepath)
        return AudioPreprocessor.wav_to(wav, sr, n_fft=n_fft, to=to, normalized=normalized, save_as=save_as)


    @staticmethod
    def wav_to(wav, sr, n_fft=2048, to='spectrogram', normalized=True, save_as=None):
        if to == 'spectrogram':
            spectrogram = AudioPreprocessor.wav_to_spectrogram(wav, sr)
        elif to == 'melspectrogram':
            spectrogram = AudioPreprocessor.wav_to_melspectrogram(wav, sr, n_fft, n_mels=64)
        else:
            raise Exception('format %s is unknown.' % to)
        
        if normalized:
            # Mean = 0, Std = 1
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
            assert spectrogram.mean() < 1e-4, spectrogram.mean()
            assert spectrogram.std() -1 < 1e-4, spectrogram.std()

        if save_as is not None:
            if not os.path.exists(os.path.dirname(save_as)): 
                os.makedirs(save_as)
            np.save(save_as, spectrogram)
        
        return spectrogram


    @staticmethod
    def detect_leading_silence(wav, silence_threshold=-50.0, chunk_size=10):
        '''
            sound is a pydub.AudioSegment
            silence_threshold in dB
            chunk_size in ms

            iterate over chunks until you find the first one with sound
        '''
        assert chunk_size > 0 # to avoid infinite loop
        
        trim_ms = 0
        while wav[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(wav):
            trim_ms += chunk_size

        return trim_ms

    @staticmethod
    def remove_silence(wav, silence_threshold=-50, chunk_size=10):
        # Detect silence at beginning and end of song
        start_trim = AudioPreprocessor.detect_leading_silence(wav, silence_threshold=silence_threshold, chunk_size=chunk_size)
        end_trim = AudioPreprocessor.detect_leading_silence(wav.reverse(), silence_threshold=silence_threshold, chunk_size=chunk_size)

        # Remove silence
        duration = len(wav)
        trimmed_sound = wav[start_trim:duration-end_trim]
        return trimmed_sound


    @staticmethod
    def split_wav_file(file_path, seconds=5, sample_type='random'):
        # Seconds to miliseconds
        miliseconds = seconds * 1000
        # Read file
        wav = pydub.AudioSegment.from_wav(file_path)
        # Remove silence at the beginning and end of wav
        wav = AudioPreprocessor.remove_silence(wav)
        # If wav file is too small ignore file
        if len(wav) < miliseconds:
            return []

        # Split file
        if sample_type == 'all':
            wavs = [wav[step: step + miliseconds] for step in range(0, len(wav)-miliseconds, miliseconds)]
        elif sample_type == 'random':
            indx = np.random.random_integers(0, len(wav)-miliseconds)
            wavs = [wav[indx: indx + miliseconds]]
        else:
            raise Exception('Sample type %s unknown' % sample_type)

        # Save temporary files
        try:
            output_names = ['temp_%d' % i for i in range(len(wavs))]
            for i, wav in enumerate(wavs):
                wav.export(output_names[i])

            return output_names
        except:
            # If any problem happens continues execution
            return []

    @staticmethod
    def visualize_spectrogram(spectrogram, sr):
        librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram')

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()

        plt.show()
