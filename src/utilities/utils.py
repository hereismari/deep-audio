import regex
import json

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