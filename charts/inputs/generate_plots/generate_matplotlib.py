#!/usr/bin/env python3

from dlcharts.common.cvlog import cvlog
from dlcharts.common.utils import swap_rb

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import time
import copy
import json
import os
from typing import Dict
from pathlib import Path

from icecream import ic

def parse_command_line():
    parser = argparse.ArgumentParser(description='Generating training images using Matplotlib')
    parser.add_argument('output_dir', help='Output folder for the generated images', type=Path)
    parser.add_argument('--debug', help='Toggle visual debugging', default=False, action='store_const', const=True)
    parser.add_argument('--num-images', help='Number of images to generate', default=2, type=int)
    args = parser.parse_args()
    return args

def image_from_fig(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img

def color_index_to_label(i):
    return i*16 + 16

transparent = (0,0,0,0)

def set_axes_color(color):
    mpl.rcParams['axes.edgecolor'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color
    mpl.rcParams['grid.color'] = color

def hex_to_color(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def to_mpl_color(rgb):
    if rgb:
        return (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, 1.0)
    else:
        return (0,0,0,0)

class Config:
    def __init__(self):
        self.backend = 'agg'
        self.axes_color = '#ff0000'
        self.bg_color = to_mpl_color((255,0,255))
        self.plots_colors = [self.axes_color] + [hex_to_color(v['color']) for v in mpl.rcParams['axes.prop_cycle']]
        self.funcs = [Func.sin(3,4), Func.sin(3,5), Func.poly([1, 0.1, 0, -0.001])]
        self.linspace_args = (-1,1,5)
        self.linewidth = 1.1
        self.xscale = 0.01
        self.yscale = 10.0
        self.xoffset = -8.0
        self.yoffset = 2.0
        self.dpi = 120
class Func:
    def __init__(self) -> None:
        pass

    def sin(scale, offset):
        return lambda x : np.sin(x * scale + offset)

    def poly(coeff):
        return np.polynomial.polynomial.Polynomial(coeff)

def generate_plot (cfg: Config):
    axes_color = cfg.plots_colors[0]
    plot_colors = cfg.plots_colors

    colors_by_label = {}
    colors_by_label[0] = cfg.bg_color
    for i, color in enumerate(plot_colors):
        colors_by_label[color_index_to_label(i)] = color

    def draw(fig, ax, colors):
        x = np.linspace(*cfg.linspace_args)
        for i, func in enumerate(cfg.funcs):
            ax.plot(x*cfg.xscale + cfg.xoffset, func(x)*cfg.yscale + cfg.yoffset, color=to_mpl_color(colors[i+1]), linewidth=cfg.linewidth)
        return image_from_fig(fig)

    w, h, dpi = 256, 256, cfg.dpi

    def create_fig():
        return plt.subplots(nrows=1,ncols=1,figsize=(w/dpi, h/dpi),dpi=dpi)

    labels_image = np.zeros((h,w), dtype=np.uint8)

    mpl.use(cfg.backend)

    # Enable the background only for the rendered image.
    mpl.rcParams['axes.facecolor'] = cfg.bg_color
    set_axes_color (axes_color)
    fig,ax = create_fig()
    rendered_im = draw (fig, ax, plot_colors)
    cvlog.image(rendered_im, 'Rendered')

    mpl.rcParams['axes.facecolor'] = to_mpl_color((255,255,255))
    fig,ax = create_fig()
    im = draw (fig, ax, [None]*len(plot_colors))
    mask2d = np.any(im != 255, axis=2)
    labels_image[mask2d] = color_index_to_label(0)
    # cvlog.image(mask2d, 'axes_mask')
    # cvlog.image (im, "Axes only")

    set_axes_color (transparent)

    for i in range(len(cfg.funcs)):
        color_idx = i+1 # first color is for axes
        fig,ax = create_fig()
        colors = [None] * len(plot_colors)
        colors[color_idx] = plot_colors[color_idx]
        im = draw (fig, ax, colors)
        mask2d = np.any(im != 255, axis=2)
        labels_image[mask2d] = color_index_to_label(color_idx)
        # cvlog.image (im, f"Visible line {i}")
        # cvlog.image(mask2d, f"mask_{i}")

    cvlog.image(labels_image, 'labels')

    jsonEntries = {}
    jsonEntries['size_cols_rows'] = [im.shape[1], im.shape[0]]
    jsonEntries['tags'] = ['matplotlib']
    jsonLabels = []
    for label, bgr_color in colors_by_label.items():
        jsonLabels.append({
            'label': label,
            'rgb_color': [bgr_color[2], bgr_color[1], bgr_color[0]],
        })
    jsonEntries['labels'] = jsonLabels

    ic(jsonEntries)
    return rendered_im, labels_image, jsonEntries

if __name__ == "__main__":

    args = parse_command_line()

    # Should be started before creating any figure.
    cvlog.enabled = True

    plt.ioff()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(args.num_images):
        config = Config()
        rendered, labels, jsonEntries = generate_plot (config)

        prefix = str(args.output_dir / f"img-{i:05d}")

        cv2.imwrite(prefix + '.rendered.png', swap_rb(rendered))
        cv2.imwrite(prefix + '.labels.png', labels)
        with open(prefix + '.json', 'w') as f:
            f.write (json.dumps(jsonEntries))

    cvlog.waitUntilWindowsAreClosed()

    # cv2.imshow ('Test Image', np.random.rand(128,128,3))
    # while True:
    #     cv2.waitKey (0)
