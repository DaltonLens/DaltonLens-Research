#!/bin/bash

# ../generate_plots/generate_matplotlib.py --num-images 1000 --width 320 --height 240 mpl/320x240 &
# ../generate_plots/generate_matplotlib.py --num-images 1000 --width 640 --height 480 mpl/640x480 &
# ../generate_plots/generate_matplotlib.py --num-images 1000 --width 1280 --height 1024 mpl/1280x1024 &

# ../generate_plots/generate_matplotlib.py --no-antialiasing --num-images 1000 --width 320 --height 240 mpl-no-aa/320x240 &
# ../generate_plots/generate_matplotlib.py --no-antialiasing --num-images 1000 --width 640 --height 480 mpl-no-aa/640x480 &
# ../generate_plots/generate_matplotlib.py --no-antialiasing --num-images 1000 --width 1280 --height 1024 mpl-no-aa/1280x1024 &

# ../generate_plots/generate_matplotlib.py --scatter --num-images 1000 --width 320 --height 240 mpl-scatter/320x240 &
# ../generate_plots/generate_matplotlib.py --scatter --num-images 1000 --width 640 --height 480 mpl-scatter/640x480 &
# ../generate_plots/generate_matplotlib.py --scatter --num-images 1000 --width 1280 --height 1024 mpl-scatter/1280x1024 &

../generate_plots/generate_opencv.py --num-drawings 1000 --width 320 --height 240 opencv-drawings/320x240 &
../generate_plots/generate_opencv.py --num-drawings 1000 --width 640 --height 480 opencv-drawings/640x480 &
../generate_plots/generate_opencv.py --num-drawings 1000 --width 1280 --height 1024 opencv-drawings/1280x1024 &

../generate_plots/generate_opencv.py --background-dir imagenette2 --num-drawings 1000 --width 320 --height 240 opencv-drawings-bg/320x240 &
../generate_plots/generate_opencv.py --background-dir imagenette2 --num-drawings 1000 --width 640 --height 480 opencv-drawings-bg/640x480 &
../generate_plots/generate_opencv.py --background-dir imagenette2 --num-drawings 1000 --width 1280 --height 1024 opencv-drawings-bg/1280x1024 &

wait
