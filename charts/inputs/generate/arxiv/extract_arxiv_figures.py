#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List
from pathlib import Path
import multiprocessing
from subprocess import run
import shutil
import numpy as np
import tempfile

import fitz

from filter_figures import filter_pdf

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from axiv articles')
    parser.add_argument('input_dir', help='Input folder with the arxiv gz files.', type=Path)
    parser.add_argument('output_dir', help='Output folder for the filtered figures.', type=Path)
    args = parser.parse_args()
    return args

def is_gray (r,g,b):
    return np.isclose(r,g) and np.isclose(g,b)

def process_gz_file (gzfile: Path, raw_pdf_dir: Path, args):
    with tempfile.TemporaryDirectory() as tmp_dirname:
        tmp_dir = Path(tmp_dirname)
        print (tmp_dir)
        run(["gunzip", "-f", "-k", gzfile])
        tar_file = gzfile.with_suffix('')
        run(["tar", "-xf", tar_file, "-C", tmp_dir])
        # run(["ls", "-lhR", tmp_dir])
        pdf_files = tmp_dir.glob('**/*.pdf')
        for pdf in pdf_files:
            pixmap = filter_pdf(pdf, need_pixmap=True)
            if not pixmap:
                continue
            outpdf = raw_pdf_dir / (tar_file.name + '_' + pdf.name)
            outpng = outpdf.with_suffix('.png')
            shutil.copy(pdf, raw_pdf_dir / (tar_file.name + '_' + pdf.name))
            pixmap.save (outpng)
        tar_file.unlink()

def main ():
    args = parse_command_line()
    gzfiles = args.input_dir.glob('**/*.gz')
    raw_pdf_dir = args.output_dir
    raw_pdf_dir.mkdir(exist_ok=True)

    with multiprocessing.Pool(8) as pool:
        results = []
        for gzfile in gzfiles:
            r = pool.apply_async (process_gz_file, (gzfile, raw_pdf_dir, args))
            results.append(r)
        for r in results:
            r.wait()

if __name__ == "__main__":
    main ()
