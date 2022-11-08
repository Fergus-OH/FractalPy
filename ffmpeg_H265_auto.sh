#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

FILENAME=$(ls -Art ./gifs | tail -n 1)

FILENAME=$(echo ${FILENAME%.*}| cut -d'_' -f -2-)

PATHSAVE="./videos/${FILENAME// /}.mp4"
echo $PATHSAVE

# Extract fps
TEMP=${FILENAME%fps*}

FPS="${TEMP##*_}"
echo $FPS


# Compile video
ffmpeg -framerate $FPS -i ./images/frames/frame%d.png  -c:v libx265 -vtag hvc1 -filter:v "scale=in_color_matrix=auto:in_range=auto:out_color_matrix=bt709:out_range=tv" -pix_fmt:v "yuv420p" -colorspace:v "bt709" -color_primaries:v "bt709" -color_trc:v "bt709" -color_range:v "tv" -c:a copy -r $FPS $PATHSAVE