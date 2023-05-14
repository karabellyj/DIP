#!/bin/bash
count=0
for motion_dir in $1/*; do
    num=$(echo $motion_dir | grep -Eo '[0-9]+$')
    echo $num
    for i in $(seq 0 3); do
        echo $motion_dir"rgb_env0_cam"$i"_frame*.png";
        ffmpeg -y -framerate 30 -pattern_type glob -i $motion_dir"/rgb_env0_cam"$i"_frame*.png"  -vcodec h264 -c:v h264 -pix_fmt yuv420p -preset slow -crf 22 -c:a aac -b:a 128k  1-$num-$i.mp4
    done
done
