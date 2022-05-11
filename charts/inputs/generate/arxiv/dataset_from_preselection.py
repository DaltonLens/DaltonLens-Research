#!/usr/bin/env python3

import argparse
from enum import Enum
import os
import re
import sys
from pathlib import Path
import shutil
import multiprocessing
from subprocess import run, DEVNULL
from xmlrpc.client import Boolean
import cv2
from dlcharts.common.utils import printBold
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from itertools import chain
import tempfile

from icecream import ic

import zv

script_path = Path(os.path.dirname(os.path.realpath(sys.argv[0])))

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from axiv articles')
    parser.add_argument('input_dir', help='Input folder with the preselected subdir', type=Path)
    parser.add_argument('--preselection_dir', help='Overwrite the source preselection dir', type=Path, default=None)
    parser.add_argument('--dataset_dir', help='Overwrite the destination selection dir', type=Path, default=None)
    args = parser.parse_args()
    return args

@dataclass 
class Rect:
    x: int
    y: int
    width: int
    height: int

def resize(imPath: Path, target_size: Tuple[int]):
    im = cv2.imread(str(imPath))
    rows = im.shape[0]
    cols = im.shape[1]
    target_cols, target_rows = target_size
    outIm = np.zeros((target_rows, target_cols, im.shape[2]), dtype=np.uint8)
    
    leftBorder = 0
    rightBorder = 0
    topBorder = 0
    bottomBorder = 0

    # Heuristic to determine if it's a white background.
    # Can be better than border replace if the image does not have a border.
    white_background = False
    num_white = np.count_nonzero (np.all(im == 255, -1))
    if num_white > (rows*cols*0.1):
        white_background = True

    if cols < target_cols:
        leftBorder = (target_cols - cols) // 2
        rightBorder = target_cols - cols - leftBorder

    if rows < target_rows:
        topBorder = (target_rows - rows) // 2
        bottomBorder = target_rows - rows - topBorder

    if (topBorder + bottomBorder + leftBorder + rightBorder) > 0:
        if white_background:
            padded_im = cv2.copyMakeBorder(im, topBorder, bottomBorder, leftBorder, rightBorder, cv2.BORDER_CONSTANT, None, (255,255,255))
        else:
            padded_im = cv2.copyMakeBorder(im, topBorder, bottomBorder, leftBorder, rightBorder, cv2.BORDER_REPLICATE, None)
    else:
        padded_im = im

    padded_rows = padded_im.shape[0]
    padded_cols = padded_im.shape[1]
    sourceOffset = [0, 0]
    if target_cols < padded_cols:
        sourceOffset[0] = (padded_cols - target_cols) // 2
    if target_rows < padded_rows:
        sourceOffset[1] = (padded_rows - target_rows) // 2

    outIm = padded_im[sourceOffset[1]:sourceOffset[1]+target_rows, sourceOffset[0]:sourceOffset[0]+target_cols, :]
    outIm = np.ascontiguousarray(outIm)
    cv2.imwrite(str(imPath), outIm)

    # zvlog.image(imPath.name, im)
    # zvlog.image(imPath.name + '_resized', outIm)
    # zvlog.waitUntilWindowsAreClosed()
    # zvlog.start()

def replace_substrings(s, spans):
    out_s = ""
    last_index = 0
    for span, v in spans:
        out_s += s[last_index:span[0]] + str(v)
        last_index = span[1]
    out_s += s[last_index:]
    return out_s

def set_svg_min_stroke_width(svg_in, svg_out, min_stroke_width=2.0):
    # Set the stroke width to at least min_stroke_width.
    # This ensures that all renderers will make the lines thicker,
    # because some renderers will make very thin lines without
    # antialiasing if the width is less than 1.
    # NOTE: this can fail if the stroke width is getting scaled with
    # For example <path transform="matrix(.1,0,0,-.1,0,364)" stroke-width="9.0"
    # will be rendered with a width of 9.0*0.1 = 0.9. To fix that we'd need to parse
    # the matrix field, and adjust the stroke width accordingly.
    with open(svg_in, 'r') as f, open(svg_out, 'w') as f_out:
        for l in f:
            spans = []
            for m in re.finditer(r'stroke-width="([0-9.]*)"', l):
                v = float(m.group(1))
                v = max(v, min_stroke_width)
                spans.append((m.span(1), v))
            if len(spans) == 0:
                f_out.write (l)
            else:
                f_out.write (replace_substrings (l, spans))

def render_pdf(pdf_file: Path, out_dir):
    svg_file_aa = out_dir / pdf_file.with_suffix('.aa.svg').name
    svg_file_aliased = out_dir / pdf_file.with_suffix('.aliased.svg').name
    pdf_text_as_path_aa = out_dir / pdf_file.with_suffix('.text_as_path_aa.pdf').name
    pdf_text_as_path_aliased = out_dir / pdf_file.with_suffix('.text_as_path_aliased.pdf').name
    out_r72_antialiased = out_dir / pdf_file.with_suffix('.r72.antialiased.png').name
    out_r72_aliased = out_dir / pdf_file.with_suffix('.r72.aliased.png').name
    out_r56_antialiased = out_dir / pdf_file.with_suffix('.r56.antialiased.png').name
    out_r56_aliased = out_dir / pdf_file.with_suffix('.r56.aliased.png').name
    run(["mutool", "draw", "-O", "text=path", "-o", svg_file_aa, pdf_file, "1"], stdout=DEVNULL, check=True)

    set_svg_min_stroke_width(svg_file_aa, svg_file_aliased)

    run(["cairosvg", svg_file_aa, "-o", pdf_text_as_path_aa], check=True)
    run(["cairosvg", svg_file_aliased, "-o", pdf_text_as_path_aliased], check=True)

    run(["gs", "-r72", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r72_antialiased}", "-dGraphicsAlphaBits=4", "-dTextAlphaBits=1", pdf_text_as_path_aa], stdout=DEVNULL, check=True)
    run(["gs", "-r72", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r72_aliased}", "-dGraphicsAlphaBits=1", "-dTextAlphaBits=1", pdf_text_as_path_aliased], stdout=DEVNULL, check=True)
    run(["gs", "-r56", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r56_antialiased}", "-dGraphicsAlphaBits=4", "-dTextAlphaBits=1", pdf_text_as_path_aa], stdout=DEVNULL, check=True)
    run(["gs", "-r56", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r56_aliased}", "-dGraphicsAlphaBits=1", "-dTextAlphaBits=1", pdf_text_as_path_aliased], stdout=DEVNULL, check=True)
    im = cv2.imread(str(out_r72_antialiased))
    rows = im.shape[0]
    cols = im.shape[1]
    valid_sizes = [(1280,960), (640,480), (320,240)]
    best_size = valid_sizes[0]
    for size in valid_sizes:
        dc = abs(cols-size[0])
        if dc < abs(cols-best_size[0]):
            best_size = size

    for im in [out_r72_antialiased, out_r72_aliased, out_r56_antialiased, out_r56_aliased]:
        resize (im, best_size)

    return best_size

class Result(Enum):
    ACCEPT = 1
    REJECT = 2
    SKIP = 3

class Validator:
    def __init__(self):
        self.app = zv.App()
        self.app.initialize()
        self.app.removeViewer("default")

    def onEvent(self, user_data):
        if zv.imgui.IsKeyPressed(zv.imgui.Key.K, False):
            self.processImage (True)
        elif zv.imgui.IsKeyPressed(zv.imgui.Key.D, False):
            self.processImage (False)

    def processImage (self, keep_it: Boolean):
        self.accepted = keep_it
        self.viewerDone = True

    def validate (self, pdf_file: Path) -> Result:
        viewer_name = pdf_file.name
        self.viewer = self.app.createViewer(pdf_file.name)
        self.current_pdf_file = pdf_file
        self.accepted = False
        def callback (*args):
            return self.onEvent(*args)
        self.viewer.setGlobalEventCallback (callback, None)
        self.viewer.setLayout(2,3)
        
        png_files = sorted(pdf_file.parent.glob(pdf_file.with_suffix('').name + '.*.png'))
        for png_file in png_files:
            self.viewer.addImageFromFile(str(png_file), False)
        self.viewerDone = False
        while (not self.viewerDone and self.app.numViewers > 0):
            self.app.updateOnce(1.0 / 30.0)
        self.app.removeViewer (viewer_name)
        if not self.viewerDone:
            return Result.SKIP
        return Result.ACCEPT if self.accepted else Result.REJECT
        

def main (args):
    # zvlog.start ()
    preselected_dir = args.input_dir / 'preselection' / 'selected' if args.preselection_dir is None else args.preselection_dir
    pdf_files = sorted(preselected_dir.glob('**/*.pdf'))

    out_dataset_dir = args.input_dir / 'dataset' if args.dataset_dir is None else args.dataset_dir
    out_discarded_dir = out_dataset_dir / 'discarded'
    out_discarded_dir.mkdir(exist_ok=True, parents=True)

    print ('out_discarded_dir', out_discarded_dir)

    validator = Validator()

    tempdir = args.input_dir / "tmp"
    tempdir.mkdir(exist_ok=True, parents=True)

    for pdf_file in pdf_files:
        # Assume validated at first.
        printBold ("Processing ", pdf_file.name)
        out_pdf = Path(tempdir) / pdf_file.name
        image_size = render_pdf (pdf_file, Path(tempdir))

        run([script_path.parent / 'gt_from_pairs' / 'build' / 'gt_from_pairs',
            out_pdf.with_suffix('.r72.antialiased.png'), 
            out_pdf.with_suffix('.r72.aliased.png'),
            out_pdf.with_suffix('.r72'),])

        run([script_path.parent / 'gt_from_pairs' / 'build' / 'gt_from_pairs',
            out_pdf.with_suffix('.r56.antialiased.png'), 
            out_pdf.with_suffix('.r56.aliased.png'),
            out_pdf.with_suffix('.r56'),])
        
        result = validator.validate (out_pdf)
        if result == Result.SKIP:
            print ("Skipped.")
            for f in tempdir.glob(out_pdf.with_suffix('').name + '.*'):
                f.unlink ()
            continue
        validated = (result == Result.ACCEPT)
        source_files = [pdf_file]
        png_file = pdf_file.with_suffix('.png')
        if png_file.exists():
            source_files.append (png_file)
        generated_files = list(out_pdf.parent.glob(out_pdf.with_suffix('').name + '.*'))
        files_to_copy = set(source_files) | set(generated_files)
        for f in files_to_copy:
            if validated:
                width, height = image_size
                target_dir = out_dataset_dir / f'{width}x{height}'
                target_dir.mkdir(exist_ok=True, parents=True)
            else:
                target_dir =out_discarded_dir
            try:
                shutil.move (str(f), str(target_dir))
            except Exception as e:
                print ('source_files:', list(source_files))
                print ('generated_files: ', generated_files)
                print ('files_to_copy: ', files_to_copy)
                raise e

if __name__ == "__main__":
    args = parse_command_line()
    main (args)
