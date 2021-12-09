IFS=$'
'
for video in $( find . | grep mp4 | grep downloaded | cut -c23- | shuf)
do
    if [ -f "kinetics_256/$video" ]
    then
        echo "already_done"
    else
ffmpeg-git-20210920-amd64-static/ffmpeg  -i "kinetics_downloaded/$video" \
    -vf scale=w=256:h=256:force_original_aspect_ratio=increase:force_divisible_by=2,crop=256:256 \
      "kinetics_256/$video"

    fi
done
