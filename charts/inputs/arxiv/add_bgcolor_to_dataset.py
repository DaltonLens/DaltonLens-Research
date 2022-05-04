#!/usr/bin/env python3

import argparse
from enum import Enum
import json
import os
import re
import sys
from pathlib import Path
import shutil
import multiprocessing
from subprocess import run, DEVNULL
from types import SimpleNamespace
from xmlrpc.client import Boolean
import cv2
from dlcharts.common.utils import printBold
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from itertools import chain
import tempfile
from types import SimpleNamespace
import random
import time

import dataset_from_preselection

import zv

script_path = Path(os.path.dirname(os.path.realpath(sys.argv[0])))

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from axiv articles')
    parser.add_argument('input_dir', help='Input folder with the dataset subdir', type=Path)
    args = parser.parse_args()
    return args

def bg_color_is_white(jsonDict):
    for labelDict in jsonDict['labels']:
        if labelDict['label'] == 0:
            return labelDict['rgb_color'] == [255, 255, 255]
    return False

def color_are_similar(color1, color2, threshold):
    for k in range(0,3):
        if abs(color1[k] - color2[k]) < threshold:
            return True
    return False

def find_different_color(jsonDict: Dict):
    existing_colors = []
    for labelDict in jsonDict['labels']:
        existing_colors.append (labelDict['rgb_color'])
    chosen_color = None
    for i in range(0, 2048):
        cand_color = [random.randint(0, 255) for k in range(0,3)]
        unique = True
        for other_color in existing_colors:
            if color_are_similar(other_color, cand_color, 30):
                unique = False
                break
        if unique:
            chosen_color = cand_color
            break
            
    return rgb_to_hex(chosen_color) if chosen_color else None

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])

def main ():
    args = parse_command_line()
    random.seed(time.time_ns())

    tempdir = args.input_dir / "tmp"
    tempdir.mkdir(exist_ok=True, parents=True)

    # zvlog.start ()
    output_dir = args.input_dir / 'dataset_bg'

    for size in ['320x240', '640x480', '1280x1024']:
        d = args.input_dir / 'dataset' / size
        json_files = sorted(d.glob('**/*.r72.json'))
    
        for json_file in json_files:
            with open(json_file, 'r') as f:
                jsonDict = json.load(f)

            if not bg_color_is_white(jsonDict):
                print ("Background is not white, skipping.")
                continue            
            
            print (f"Processing {json_file}")

            svg_file_aa = json_file.parent / (json_file.stem.replace('.r72','') + '.aa.svg')
            if not svg_file_aa.exists():
                print (f"Error: could not find {svg_file_aa}")
                continue
            
            # Always flush the temp dir before processing a new guy to avoid
            # previous datasets to creep in.
            for f in tempdir.iterdir():
                f.unlink()

            out_svg = tempdir / svg_file_aa.name.replace('.aa.','.bg.')
            out_pdf = out_svg.with_suffix('.pdf')
            print('out_pdf', out_pdf)

            if (output_dir / size / out_pdf.name).exists():
                print ("Was already accepted, skipping.")
                continue

            if (output_dir / 'discarded' / out_pdf.name).exists():
                print ("Was already discarded, skipping.")
                continue

            new_bg_color = find_different_color(jsonDict)
            if new_bg_color is None:
                print ("Could not find a compatible color, skipping.")
                continue
            print (new_bg_color)

            with open(svg_file_aa, 'r') as f_in, open(out_svg, 'w') as f_out:
                for l in f_in:
                    f_out.write(l.replace ('fill="#ffffff"', f'fill="{new_bg_color}"'))
                    if l.startswith('<g enable-background'):
                        f_out.write(f'<rect width="200%" height="200%" fill="{new_bg_color}"/>\n')

            run(["cairosvg", out_svg, "-o", out_pdf])
            out_svg.unlink ()

            val_args = SimpleNamespace()
            val_args.input_dir = args.input_dir
            val_args.dataset_dir = output_dir
            val_args.preselection_dir = tempdir
            dataset_from_preselection.main (val_args)

if __name__ == "__main__":
    main ()
