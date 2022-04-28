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
from typing import Tuple
from dataclasses import dataclass
from itertools import chain
import tempfile
from types import SimpleNamespace

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

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])

def main ():
    args = parse_command_line()
    
    tempdir = args.input_dir / "tmp"
    tempdir.mkdir(exist_ok=True, parents=True)

    # zvlog.start ()
    default_bg_dir = args.input_dir / 'dataset' / 'validated'
    output_dir = args.input_dir / 'dataset_bg'

    json_files = sorted(default_bg_dir.glob('**/*.r72.json'))
    for json_file in json_files:
        with open(json_file, 'r') as f:
            jsonDict = json.load(f)

        if not bg_color_is_white(jsonDict):
            print ("Background is not white, skipping.")
            continue            

        if (output_dir / 'validated' / json_file.with_suffix('.pdf')).exists():
            continue

        if (output_dir / 'discarded' / json_file.with_suffix('.pdf')).exists():
            continue
        
        print (f"Processing {json_file}")

        svg_file_aa = json_file.parent / (json_file.stem.replace('.r72','') + '.aa.svg')
        if not svg_file_aa.exists():
            print (f"Error: could not find {svg_file_aa}")
            continue
        
        out_svg = tempdir / svg_file_aa.name
        new_bg_color = rgb_to_hex([255, 255, 0])
        print (new_bg_color)

        with open(svg_file_aa, 'r') as f_in, open(out_svg, 'w') as f_out:
            for l in f_in:
                f_out.write(l.replace ('fill="#ffffff"', f'fill="{new_bg_color}"'))
                if l.startswith('<g enable-background'):
                    f_out.write(f'<rect width="100%" height="100%" fill="{new_bg_color}"/>\n')

        run(["cairosvg", out_svg, "-o", out_svg.with_suffix('.pdf')])
        out_svg.unlink ()

        val_args = SimpleNamespace()
        val_args.input_dir = args.input_dir
        val_args.dataset_dir = output_dir
        val_args.preselection_dir = tempdir
        dataset_from_preselection.main (val_args)

        break

if __name__ == "__main__":
    main ()
