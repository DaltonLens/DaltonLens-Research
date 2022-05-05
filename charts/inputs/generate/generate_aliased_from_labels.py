#!/usr/bin/env python3

import argparse, json, os, random, sys
from pathlib import Path

from dlcharts.common.dataset import LabeledImage

import cv2
from dlcharts.common.utils import swap_rb, debug

from zv.log import zvlog

if debug:
    zvlog.start()

parser = argparse.ArgumentParser(description='Generate aliased rgb images from GT labels.')
parser.add_argument('json_files', type=Path, help='Json files with the labels', nargs='+')
args = parser.parse_args()

for json_file in args.json_files:
    im = LabeledImage(json_file)
    im.compute_labels_as_rgb ()
    cv2.imwrite(str(im.json_file.with_suffix('.aliased.png')), swap_rb(im.labels_as_rgb))

if debug:
    zvlog.waitUntilWindowsAreClosed()