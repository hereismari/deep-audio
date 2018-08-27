if ! [[ -d "$1" ]]
then
    echo "The first argument must be a directory."
else
    echo "Spliting wav files at $1 in 10 seconds pieces. Results can be found on folder $1/splitted"
    mkdir "$1/splitted"
    for file in "$1"/*.wav; do
        ffmpeg -i "$file" -c copy -map 0 -segment_time $2 -f segment "$1/splitted/$(basename "$file" .wav)_%03d.wav"
    done
fi