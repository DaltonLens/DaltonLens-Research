#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import cv2

def parse_command_line():
    parser = argparse.ArgumentParser(description='Upsample the resolution of a folder')
    parser.add_argument('input_dir', help='Input folder with the dataset subdir', type=Path)
    parser.add_argument('output_dir', help='Output folder for the upsampled dataset', type=Path)
    args = parser.parse_args()
    return args

def upsample (in_png: Path, out_png: Path, interpolation):
    img = cv2.imread(str(in_png))
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=interpolation)
    cv2.imwrite(str(out_png), img)

def main ():
    args = parse_command_line()
    json_files = args.input_dir.glob('**/*.json')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for json_file in json_files:
        print ('.', end='', flush=True)
        aliased_png = json_file.with_suffix('.aliased.png')
        antialiased_png = json_file.with_suffix('.antialiased.png')
        labels_png = json_file.with_suffix('.labels.png')

        shutil.copyfile (json_file, args.output_dir / json_file.name)
        upsample (aliased_png, args.output_dir / aliased_png.name, cv2.INTER_NEAREST)
        upsample (labels_png, args.output_dir / labels_png.name, cv2.INTER_NEAREST)
        upsample (antialiased_png, args.output_dir / antialiased_png.name, cv2.INTER_CUBIC)

if __name__ == "__main__":
    main ()
