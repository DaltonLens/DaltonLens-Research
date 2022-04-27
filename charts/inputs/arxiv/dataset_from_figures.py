#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import shutil
import multiprocessing
from subprocess import run

import zv

script_path = Path(os.path.dirname(os.path.realpath(sys.argv[0])))

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from axiv articles')
    parser.add_argument('input_dir', help='Input folder with the selected pdfs.', type=Path)
    parser.add_argument('output_dir', help='Output dataset folder.', type=Path)
    args = parser.parse_args()
    return args

def main ():
    args = parse_command_line()
    pdf_files = args.input_dir.glob('**/*.pdf')
    args.output_dir.mkdir(exist_ok=True, parents=True)
    for pdf_file in pdf_files:
        out_pdf = args.output_dir / pdf_file.name
        shutil.copy(pdf_file, out_pdf)
        run([script_path / "render_pdf.sh", out_pdf])
        
        run([script_path.parent / 'generate_plots' / 'gt_from_pairs' / 'build' / 'gt_from_pairs',
             out_pdf.with_suffix('.pdf.r72.antialiased.png'), 
             out_pdf.with_suffix('.pdf.r72.aliased.png'),
             out_pdf.with_suffix('.pdf.r72'),])

        run([script_path.parent / 'generate_plots' / 'gt_from_pairs' / 'build' / 'gt_from_pairs',
             out_pdf.with_suffix('.pdf.r56.antialiased.png'), 
             out_pdf.with_suffix('.pdf.r56.aliased.png'),
             out_pdf.with_suffix('.pdf.r56'),])
        
        print ("Finished processing ", out_pdf)

if __name__ == "__main__":
    main ()
