from __future__ import unicode_literals
import commands
import argparse

import utilities.utils as utils
import numpy as np


parser = argparse.ArgumentParser('')

parser.add_argument('--run-mode', type=str, default='playlist-file',
                    choices=['playlist-file', 'video-list', 'playlist-list'])
parser.add_argument('--output-path', type=str, default='audio_files/')

parser.add_argument('--playlist-file', type=str, default='download_data/conf/playlists.json')
parser.add_argument('--songs-per-playlist', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--youtube-conf', type=str, default='download_data/conf/youtube.json')


def download_youtube_audio(output_path, playlist_file):
    # Loads laylists
    with open("./playlists.txt", "r") as f:
        playlists = []
        for line in f:
            playlists.append(line)
        playlists = [x.strip() for x in playlists] 


    for i in range(0, int(nbVideos)):
        print("Download " + repr(i))
        try:
            playlistInd = randint(1, 500)
            commands.getstatusoutput("youtube-dl \
                --ignore-errors \
                --audio-quality 0 \
                -x \
                -o '" + os.path.join(output_path, + repr(i) + ".%(ext)s' \
                --audio-format wav \
                --playlist-start " + repr(playlistInd) + " --playlist-end " + repr(playlistInd) + " "\
                + playlists[i % len(playlists)]))
        except:
            print("Error during download")



def download_from_playlist_file(playlist_file, songs_per_laylist, output_path):
    playlists_urls = utils.load_json(playlist_file)
    for playlist in playlists_urls:
        id = utils.id_from_youtube_url(playlists_urls[playlist])
        indices = np.random.choice(range(songs_per_laylist)), replace=False)
        




def download_from_playlist_list():
    pass


def download_from_video_list():
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if args.run_mode == 'playlist-file':
        download_from_playlist_file()
    elif args.run_mode == 'playlist-list':
        download_from_playlist_list()
    elif args.run_mode == 'video-list':
        download_from_video_list()