'''
Author: Marianne Linhares, mariannelinharesm@gmail.com
Last Change: 08/2018

Based on the script from BGodefroyFR
at: https://github.com/BGodefroyFR/Deep-Audio-Visualization/blob/master/learning/scripts/downloadTrainingSamples.py
'''
import argparse

import matplotlib.pyplot as plt
import librosa
import scipy

import numpy as np
import os
import glob


parser = argparse.ArgumentParser('')

parser.add_argument('--run-mode', type=str, default='path',
                    choices=['file', 'path'])

parser.add_argument('--output-path', type=str, default='audio_files/preprocessed.npy')
parser.add_argument('--input-path', type=str, default='audio_files/splitted/')
parser.add_argument('--n_fft', type=int, default=2048)


def wav_to_spectrogram(input_path, n_fft, save_file=False, output_path=None):
    '''Code from: https://github.com/DmitryUlyanov/neural-style-audio-tf/blob/master/neural-style-audio-tf.ipynb'''
    # Load wav file
    y, sr = librosa.load(input_path)
    print(input_path)
    
    # Generate spectrogram
    spectrogram = librosa.stft(y, n_fft, window=scipy.signal.hamming)
    
    mel_spectrogram, mel_basis, S = melspectrogram(y, n_mels=64)
    spectrogram = np.log1p(np.abs(spectrogram[:, :257]))

    # Save spectrogram
    if save_file:
        if not os.path.exists(os.path.dirname(output_path)): 
            os.makedirs(output_path)
        np.save(output_path, spectrogram)
    
    return mel_spectrogram, sr


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


def every_wav_to_spectrogram(input_path, output_path, n_fft):
    if not os.path.exists(os.path.dirname(output_path)): 
        os.makedirs(output_path)

    spectrograms = []
    for i, f in enumerate(glob.glob(os.path.join(input_path,'*'))):
        if i % 100 == 0 and i > 0:
            break
            print('Preprocessing %d ...' % i)
        spectrogram, sr = wav_to_spectrogram(f, n_fft)
        if spectrogram.shape != (64, 216):
            continue
        spectrograms.append(spectrogram)

    np.save(output_path, spectrograms)
    return spectrograms


def visualize_spectrogram(spectrogram):
    plt.imshow(spectrogram), plt.pause(2)


def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):
    S, n_fft = librosa.core.spectrum._spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=power)
    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr, n_fft, **kwargs)
    return np.dot(mel_basis, S), mel_basis, S


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.run_mode == 'file':
        wav_to_spectrogram(args.input_path, args.n_fft, save_file=True, output_path=args.output_path)
    elif args.run_mode == 'path':
        every_wav_to_spectrogram(args.input_path, args.output_path, args.n_fft)