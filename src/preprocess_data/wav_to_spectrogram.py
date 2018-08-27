'''
Author: Marianne Linhares, mariannelinharesm@gmail.com
Last Change: 08/2018

Based on the script from BGodefroyFR
at: https://github.com/BGodefroyFR/Deep-Audio-Visualization/blob/master/learning/scripts/downloadTrainingSamples.py
'''
import argparse

import matplotlib.pyplot as plt
import librosa
import librosa.display

import numpy as np
import os


parser = argparse.ArgumentParser('')

parser.add_argument('--run-mode', type=str, default='file',
                    choices=['file', 'path'])

parser.add_argument('--output-path', type=str, default='preprocessed_files/')
parser.add_argument('--input-path', type=str, default='audio_files/classical_145.wav')
parser.add_argument('--window-size', type=int, default=11025)


def download_from_playlist_file(playlist_file, songs_per_playlist, output_path):
    playlists_urls = utils.load_json(playlist_file)
    for playlist in playlists_urls:
        playlist_id = utils.id_from_youtube_url(playlists_urls[playlist]['url'])
        length = playlists_urls[playlist]['length']
        indices = np.random.choice(range(length), size=min(length, songs_per_playlist), replace=False)

        for index in indices:
            print("Downloading song %s from playlist %s" % (index, playlist))
            final_filename = os.path.join(output_path,  '%s_%s' % (playlist, index))
            output_filename = '"' + final_filename + '.%(ext)s"'
            utils.mkdir_if_needed(os.path.dirname(final_filename))

            if os.path.exists(final_filename + '.wav'):
                continue

            try:
                command = ('youtube-dl \
                    --extract-audio \
                    --ignore-errors \
                    --audio-quality 0 \
                    -x \
                    -o ' + output_filename + ' --audio-format wav \
                    --playlist-start ' + str(index) + ' --playlist-end ' + str(index) + ' ' \
                    + playlist_id)
                utils.get_status_output(command)
            except:
                print("Error during download :(")

def wav_to_spectrogram(input_path, output_path, window_size):
    y, sr = librosa.load(input_path)
    number_of_windows = int(len(y)/window_size)
    spectrogram = np.empty((number_of_windows, window_size))

    for i in range(number_of_windows):
        spectrogram[i, :] = abs(np.fft.fft(y[i * window_size: (i+1) * window_size]))
    
    melgram = librosa.power_to_db(librosa.feature.melspectrogram(y, sr=sr, n_mels=128))
    # melgram = melgram / np.linalg.norm(melgram)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(melgram,
                             y_axis='mel', fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show(), plt.pause(10)
    return melgram


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.run_mode == 'file':
        wav_to_spectrogram(args.input_path, args.output_path, args.window_size)
    elif args.run_mode == 'path':
        download_from_playlist_list()