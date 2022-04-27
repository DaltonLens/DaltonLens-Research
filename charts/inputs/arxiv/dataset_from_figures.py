#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import shutil
import multiprocessing
from subprocess import run
import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass

from zv.log import zvlog

script_path = Path(os.path.dirname(os.path.realpath(sys.argv[0])))

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from axiv articles')
    parser.add_argument('input_dir', help='Input folder with the selected pdfs.', type=Path)
    parser.add_argument('output_dir', help='Output dataset folder.', type=Path)
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

    if cols < target_cols:
        leftBorder = (target_cols - cols) // 2
        rightBorder = target_cols - cols - leftBorder

    if rows < target_rows:
        topBorder = (target_rows - rows) // 2
        bottomBorder = target_rows - rows - topBorder

    if (topBorder + bottomBorder + leftBorder + rightBorder) > 0:
        padded_im = cv2.copyMakeBorder(im, topBorder, bottomBorder, leftBorder, rightBorder, cv2.BORDER_REPLICATE, None, None)
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

def render_pdf(pdf_file: Path):
    svg_file = pdf_file.with_suffix('.svg')
    pdf_text_as_path = pdf_file.with_suffix('.text_as_path.pdf')
    file_prefix = pdf_file.stem
    out_r72_antialiased = pdf_file.with_suffix('.r72.antialiased.png')
    out_r72_aliased = pdf_file.with_suffix('.r72.aliased.png')
    out_r56_antialiased = pdf_file.with_suffix('.r56.antialiased.png')
    out_r56_aliased = pdf_file.with_suffix('.r56.aliased.png')
    run(["mutool", "draw", "-O", "text=path", "-o", svg_file, pdf_file, "1"])
    run(["cairosvg", svg_file, "-o", pdf_text_as_path])
    run(["gs", "-r72", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r72_antialiased}", "-dGraphicsAlphaBits=4", "-dTextAlphaBits=1", pdf_text_as_path])
    run(["gs", "-r72", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r72_aliased}", "-dGraphicsAlphaBits=1", "-dTextAlphaBits=1", pdf_text_as_path])
    run(["gs", "-r56", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r56_antialiased}", "-dGraphicsAlphaBits=4", "-dTextAlphaBits=1", pdf_text_as_path])
    run(["gs", "-r56", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-sOutputFile={out_r56_aliased}", "-dGraphicsAlphaBits=1", "-dTextAlphaBits=1", pdf_text_as_path])
    im = cv2.imread(str(out_r72_antialiased))
    rows = im.shape[0]
    cols = im.shape[1]
    valid_sizes = [(1280,1024), (640,480), (320,240)]
    best_size = valid_sizes[0]
    for size in valid_sizes:
        dc = abs(cols-size[0])
        if dc < abs(cols-best_size[0]):
            best_size = size

    for im in [out_r72_antialiased, out_r72_aliased, out_r56_antialiased, out_r56_aliased]:
        resize (im, best_size)

def main ():
    args = parse_command_line()
    # zvlog.start ()
    pdf_files = args.input_dir.glob('**/*.pdf')
    args.output_dir.mkdir(exist_ok=True, parents=True)
    for pdf_file in pdf_files:
        out_pdf = args.output_dir / pdf_file.name
        if out_pdf.exists():
            print ("Skipping", pdf_file.name)
            continue

        shutil.copy(pdf_file, out_pdf)
        render_pdf (out_pdf)
        
        print ("Rendered ", out_pdf)

        run([script_path.parent / 'generate_plots' / 'gt_from_pairs' / 'build' / 'gt_from_pairs',
             out_pdf.with_suffix('.r72.antialiased.png'), 
             out_pdf.with_suffix('.r72.aliased.png'),
             out_pdf.with_suffix('.r72'),])

        run([script_path.parent / 'generate_plots' / 'gt_from_pairs' / 'build' / 'gt_from_pairs',
             out_pdf.with_suffix('.r56.antialiased.png'), 
             out_pdf.with_suffix('.r56.aliased.png'),
             out_pdf.with_suffix('.r56'),])
        
        print ("Finished processing ", out_pdf)

if __name__ == "__main__":
    main ()
