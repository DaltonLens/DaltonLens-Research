#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import shutil
from dlcharts.common.utils import printBold
import numpy as np
import cv2

import zv

def parse_command_line():
    parser = argparse.ArgumentParser(description='Extract relevant figures from arxiv articles')
    parser.add_argument('input_dir', help='Input folder with the raw_figures subfolder.', type=Path)
    args = parser.parse_args()
    return args

class App:
    def __init__(self, args):
        self.args = args
        self.app = zv.App()
        self.raw_fig_dir = args.input_dir / 'raw_figures'
        assert self.raw_fig_dir.is_dir()

        self.out_selected_dir = args.input_dir / 'preselection' / 'selected'
        self.out_discarded_dir = args.input_dir / 'preselection' / 'discarded'
        self.out_selected_dir.mkdir(parents=True, exist_ok=True)
        self.out_discarded_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([str(p) for p in self.raw_fig_dir.glob('**/*.png')])
        print (f"Number of files: {len(files)}")
        self.app.initialize(["zv"] + files)
        self.viewer = self.app.getViewer()
        def callback (*args):
            return self.onEvent(*args)
        self.viewer.setGlobalEventCallback (callback, None)

        printBold ("Hit 'k' to keep the image, 'd' to discard it.")

    def onEvent(self, user_data):
        if zv.imgui.IsKeyPressed(zv.imgui.Key.K, False):
            self.processImage (True)
        elif zv.imgui.IsKeyPressed(zv.imgui.Key.D, False):
            self.processImage (False)

    def processImage(self, keep_it):
        image_id = self.viewer.selectedImage
        image_item = self.viewer.getImageItem (image_id)
        image_file = Path(image_item.sourceImagePath)
        in_pdf = image_file.with_suffix('.pdf')
        out_dir = self.out_selected_dir if keep_it else self.out_discarded_dir
        out_png = out_dir / image_file.name
        out_pdf = out_png.with_suffix('.pdf')
        shutil.move(image_file, out_png)
        shutil.move(in_pdf, out_pdf)
        print (f'Moved {image_file} to {out_dir}')
        self.viewer.runAction (zv.ImageWindowAction.View_NextImage)

    def run (self):
        while self.app.numViewers > 0:
            self.app.updateOnce(1.0 / 30.0)

def main ():
    args = parse_command_line()
    app = App(args)
    app.run ()

if __name__ == "__main__":
    main ()
