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

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from axiv articles')
    parser.add_argument('input_dir', help='Input folder with the arxiv gz files.', type=Path)
    parser.add_argument('output_dir', help='Output folder for the filtered figures.', type=Path)
    args = parser.parse_args()
    return args

def is_gray (r,g,b):
    return np.isclose(r,g) and np.isclose(g,b)

def filter_pdf(pdf_file):
    try:
        doc = fitz.open (pdf_file)
    except:
        return None
    if doc.page_count != 1:
        return None
    box = doc.page_cropbox(0)
    w, h = box.width, box.height
    if (w < 200 or w > 1600):
        return None
    page = doc.load_page(0)
    drawings = page.get_cdrawings()
    if len(drawings) < 5 or len(drawings) > 4096:
        return None
    colors = set()
    for d in drawings:
        if 'color' in d:
            color = d['color']
            if not is_gray(*color):
                colors.add(str(color))
    if len(colors) < 3 or len(colors) > 64:
        return None
    print (f"{pdf_file} #{len(drawings)}")
    return page.get_pixmap()

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
            pixmap = filter_pdf(pdf)
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
