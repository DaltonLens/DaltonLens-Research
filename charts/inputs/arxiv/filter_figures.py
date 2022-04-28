#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List
from pathlib import Path
import multiprocessing
from subprocess import run
import shutil
from xmlrpc.client import Boolean
import numpy as np
import tempfile
import sys

import fitz

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from axiv articles')
    parser.add_argument('input_dir', help='Input folder with the pdf files.', type=Path)
    parser.add_argument('--yes', action='store_true', help='Accept all filter output.')
    args = parser.parse_args()
    return args

def is_gray (r,g,b):
    return np.isclose(r,g) and np.isclose(g,b)

def filter_pdf(pdf_file: Path, need_pixmap: Boolean):
    try:
        doc = fitz.open (pdf_file)
    except:
        sys.stderr.write("Could not parse\n")
        return None
    if doc.page_count != 1:
        sys.stderr.write("More than 1 page\n")
        return None
    box = doc.page_cropbox(0)
    w, h = box.width, box.height
    if (w < 200 or w > 1600):
        sys.stderr.write(f"Too large w={w}\n")
        return None
    page = doc.load_page(0)
    drawings = page.get_cdrawings()
    num_drawings = len(drawings)
    if num_drawings < 5 or num_drawings > 4096:
        sys.stderr.write(f"Num drawings out of range {num_drawings}\n")
        return None
    colors = set()
    num_items = 0
    for d in drawings:
        if 'items' in d:
            num_items += len(d['items'])
        if 'color' in d:
            color = d['color']
            if not is_gray(*color):
                colors.add(str(color))
    # Some pdfs has 1M items.
    if num_items > 64000:
        sys.stderr.write(f"Too many items {num_items}\n")
        return None
    num_colors = len(colors)
    if num_colors < 3 or num_colors > 64:
        sys.stderr.write(f"Num colors out of range {num_colors}\n")
        return None
    print (f"{pdf_file} #{num_drawings}")
    return page.get_pixmap() if need_pixmap else True

def ask_for_confirmation(question: str, pdf_file: Path):
    while True:
        sys.stdout.write(question + ' (y/n/s) ')
        choice = input().lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        elif choice == 's':
            run(["xdg-open", pdf_file])
        else:
            sys.stdout.write("Please respond with 'y' or 'n'\n")

def main ():
    args = parse_command_line()
    pdf_files = args.input_dir.glob('*.pdf')
    for pdf_file in pdf_files:
        if filter_pdf(pdf_file, need_pixmap=False) is None:
            if args.yes or ask_for_confirmation (f"Remove {pdf_file}?", pdf_file):
                for f in pdf_file.parent.glob(pdf_file.with_suffix('').name + '*'):
                    f.unlink ()

if __name__ == "__main__":
    main ()
