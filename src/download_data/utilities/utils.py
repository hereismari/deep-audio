import regex
import json
import os
import subprocess


def id_from_youtube_url(url):
    match = regex.match('^(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?.*?(?:v|list)=(.*?)(?:&|$)|^(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?(?:(?!=).)*\/(.*)$',
                       url)
    if match is not None and match[1] is not None:
        return match[1]
    else:
        raise Exception('Invalid youtube url %s' % url)


def load_json(filename):
    f = open(filename)
    return json.load(f)


def mkdir_if_needed(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_status_output(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    p.wait()
    return p.returncode