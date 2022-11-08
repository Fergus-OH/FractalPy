#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo Enter framerate for output:
read FPS

echo Enter video output name with extension:
read FILENAME

ffmpeg -framerate $FPS -i ./images/frames/frame%d.png  -c:v libx265 -vtag hvc1 -filter:v "scale=in_color_matrix=auto:in_range=auto:out_color_matrix=bt709:out_range=tv" -pix_fmt:v "yuv420p" -colorspace:v "bt709" -color_primaries:v "bt709" -color_trc:v "bt709" -color_range:v "tv" -c:a copy -r $FPS ./videos/$FILENAME