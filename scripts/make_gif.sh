#!/bin/bash
# usage: bash scripts/make_gif.sh left.mp4 right.mp4 assets/out.gif ["left caption"] ["right caption"]
# Puts two rollout videos side by side as a looping GIF, 300 px tall each, with optional captions.
# The shorter video (a success ends early) holds its last frame until the longer one finishes.
# Every 3rd sim step is kept and played at 15 fps, so a 300-step episode loops in ~7 s.
set -e
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
printf '%s' "${4:-}" > "$TMP/l.txt"; printf '%s' "${5:-}" > "$TMP/r.txt"
FONT=$(fc-match -f '%{file}' sans)
CAP="drawbox=y=0:w=iw:h=34:color=black@0.65:t=fill,drawtext=fontfile=$FONT:fontsize=12:fontcolor=white:x=6:y=11"
ffmpeg -loglevel error -y -i "$1" -i "$2" -filter_complex \
 "[0:v]tpad=stop_mode=clone:stop=-1,scale=-1:300,$CAP:textfile=$TMP/l.txt[a];\
  [1:v]tpad=stop_mode=clone:stop=-1,scale=-1:300,$CAP:textfile=$TMP/r.txt[b];\
  [a][b]hstack=shortest=0,trim=end_frame=300,select='not(mod(n\,3))',setpts=N/15/TB,split[s0][s1];\
  [s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" \
 -r 15 "$3"
