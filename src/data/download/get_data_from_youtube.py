'''
Author: Marianne Linhares, mariannelinharesm@gmail.com
Last Change: 08/2018

Based on the script from BGodefroyFR
at: https://github.com/BGodefroyFR/Deep-Audio-Visualization/blob/master/learning/scripts/downloadTrainingSamples.py
'''

from __future__ import unicode_literals
import argparse

import utilities.utils as utils
import numpy as np
import os


parser = argparse.ArgumentParser('')

parser.add_argument('--run-mode', type=str, default='playlist-file',
                    choices=['playlist-file', 'video-list', 'playlist-list'])
parser.add_argument('--output-path', type=str, default='audio_files/')

parser.add_argument('--playlist-file', type=str, default='download_data/conf/playlists.json')
parser.add_argument('--songs-per-playlist', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--youtube-conf', type=str, default='download_data/conf/youtube.json')


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


def download_from_playlist_list():
    pass


def download_from_video_list():
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if args.run_mode == 'playlist-file':
        download_from_playlist_file(args.playlist_file, args.songs_per_playlist, args.output_path)
    elif args.run_mode == 'playlist-list':
        download_from_playlist_list()
    elif args.run_mode == 'video-list':
        download_from_video_list()