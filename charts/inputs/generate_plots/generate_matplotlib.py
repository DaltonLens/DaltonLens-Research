#!/usr/bin/env python3

import zv
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
from typing import Dict, List
from pathlib import Path

from icecream import ic
from tqdm import tqdm

zvApp = None
viewer = None

rgen = np.random.default_rng(int(time.time()*1e3))

def parse_command_line():
    parser = argparse.ArgumentParser(description='Generating training images using Matplotlib')
    parser.add_argument('output_dir', help='Output folder for the generated images', type=Path)
    parser.add_argument('--debug', help='Toggle visual debugging', default=False, action='store_const', const=True)
    parser.add_argument('--num-images', help='Number of images to generate', default=2, type=int)
    parser.add_argument('--margin', help='Mask extra margin', default=3, type=int)
    parser.add_argument('--no-antialiasing', help='Disable anti-aliasing', action='store_true')
    parser.add_argument('--scatter', help='Generate scatter plots', action='store_true')
    args = parser.parse_args()
    return args

def image_from_fig(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img

def color_index_to_label(i):
    return i*16 + 16

def random_rgb(often_grayscale = False):
    if often_grayscale and rgen.uniform(0,1) < 0.5:
        v = rgen.integers(0,256,dtype=int)
        return (v,v,v)
    else:
        return tuple(rgen.integers(0, 256, size=3, dtype=int).tolist())

def random_rgb_different(other_rgbs: List, often_grayscale = False):
    while True:
        candidate_rgb = random_rgb(often_grayscale)
        is_good = True
        for other in other_rgbs:
            if (np.sum(np.abs(np.subtract(candidate_rgb, other))) < 50):
                is_good = False
                break
        if is_good:
            return candidate_rgb

def random_rgb_set(n: int, prev_colors = [], often_grayscale=False):
    l = prev_colors.copy()
    for i in range(n):
        l.append(random_rgb_different(l, often_grayscale))
    return l[len(prev_colors):]

transparent = (0,0,0,0)

def set_axes_color(color):
    mpl_color = to_mpl_color(color)
    mpl.rcParams['axes.edgecolor'] = mpl_color
    mpl.rcParams['axes.labelcolor'] = mpl_color
    mpl.rcParams['xtick.color'] = mpl_color
    mpl.rcParams['ytick.color'] = mpl_color
    mpl.rcParams['grid.color'] = mpl_color

def set_bg_color(color):
    mpl_color = to_mpl_color(color)
    mpl.rcParams['figure.facecolor'] = mpl_color
    mpl.rcParams['axes.facecolor'] = mpl_color

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
        self.scatter = False
        nfuncs = rgen.integers(2,6)
        often_gray_colors = random_rgb_set(2, [], often_grayscale=True)
        self.bg_color = often_gray_colors[0]
        self.axes_color = often_gray_colors[1]
        self.plots_colors = [self.axes_color] + random_rgb_set(nfuncs, often_gray_colors)
        #  [self.axes_color] + [hex_to_color(v['color']) for v in mpl.rcParams['axes.prop_cycle']]        
        self.funcs = [Config.random_func() for v in range(nfuncs)]
        self.linspace_args = (-1,1,rgen.integers(5,100))
        
        # linewidth or scatter marker area
        self.linewidth = rgen.uniform(0.5,1.5) if rgen.uniform() < 0.9 else rgen.uniform(1.5,5)

        markers = {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase'}
        markers = list(markers.keys())
        self.markers = [markers[rgen.integers(0, len(markers))] for _ in range(nfuncs)]

        self.xscale = rgen.uniform(0.01, 100.0)
        self.yscale = rgen.uniform(0.01, 100.0)
        self.xoffset = rgen.uniform(-2, 2)
        self.yoffset = rgen.uniform(-2, 2)
        self.dpi = rgen.integers(50,150)

    def random_func():
        kind = rgen.integers(0,2)
        if kind == 0:
            return Func.sin(rgen.uniform(-5,5), rgen.uniform(-5,5))
        else:
            num_coeffs = rgen.integers(1,4)
            coeffs = rgen.uniform(-0.1, 0.1, size=num_coeffs)
            return Func.poly(coeffs)
class Func:
    def __init__(self) -> None:
        pass

    def sin(scale, offset):
        return lambda x : np.sin(x * scale + offset)

    def poly(coeff):
        return np.polynomial.polynomial.Polynomial(coeff)

def generate_plot (cfg: Config, on_rendered_image_event):
    axes_color = cfg.plots_colors[0]
    plot_colors = cfg.plots_colors

    colors_by_label = {}
    colors_by_label[0] = cfg.bg_color
    for i, color in enumerate(plot_colors):
        colors_by_label[color_index_to_label(i)] = color

    w, h, dpi = 256, 256, cfg.dpi

    def draw(fig, ax, colors):
        x = np.linspace(*cfg.linspace_args)
        for i, func in enumerate(cfg.funcs):
            if cfg.scatter:
                ax.scatter(x*cfg.xscale + cfg.xoffset, func(x)*cfg.yscale + cfg.yoffset,
                           marker=cfg.markers[i],
                           color=to_mpl_color(colors[i+1]),
                           s=cfg.linewidth)
            else:
                ax.plot(x*cfg.xscale + cfg.xoffset, func(x)*cfg.yscale + cfg.yoffset, color=to_mpl_color(colors[i+1]), linewidth=cfg.linewidth)
        im = image_from_fig(fig)
        assert im.shape[0] == h and im.shape[1] == w
        return im

    def create_fig():
        # Add a small eps to make sure that we always get the right output size.
        return plt.subplots(nrows=1,ncols=1,figsize=((w+1e-3)/dpi, (h+1e-3)/dpi),dpi=dpi)

    labels_image = np.zeros((h,w), dtype=np.uint8)

    mpl.use(cfg.backend)

    # Enable the background only for the rendered image.
    set_bg_color(cfg.bg_color)
    set_axes_color (axes_color)
    fig,ax = create_fig()
    rendered_im = draw (fig, ax, plot_colors)
    if viewer:
        image_id = viewer.addImage ('Rendered', rendered_im)
        viewer.setEventCallback (image_id, on_rendered_image_event, None)
    plt.close(fig)

    set_bg_color((255,255,255))
    fig,ax = create_fig()
    im = draw (fig, ax, [None]*len(plot_colors))
    mask2d = np.any(im < 255-args.margin, axis=2)
    labels_image[mask2d] = color_index_to_label(0)
    plt.close(fig)
    # zvlog.image('axes_mask', mask2d)
    # zvlog.image ("Axes only", im)

    # Same as the fake white background
    set_axes_color ((255,255,255))

    for i in range(len(cfg.funcs)):
        color_idx = i+1 # first color is for axes
        fig,ax = create_fig()        
        colors = [None] * len(plot_colors)
        colors[color_idx] = plot_colors[color_idx]
        im = draw (fig, ax, colors)
        mask2d = np.any(im < 255-args.margin, axis=2)
        labels_image[mask2d] = color_index_to_label(color_idx)
        plt.close(fig)
        # zvlog.image (f"Visible line {i}", im)
        # zvlog.image(f"mask_{i}", mask2d)

    if viewer:
        viewer.addImage('labels', labels_image)

    jsonEntries = {}
    jsonEntries['size_cols_rows'] = [im.shape[1], im.shape[0]]
    jsonEntries['tags'] = ['matplotlib']
    jsonLabels = []
    for label, rgb_color in colors_by_label.items():
        jsonLabels.append({
            'label': label,
            'rgb_color': [rgb_color[0], rgb_color[1], rgb_color[2]],
        })
    jsonEntries['labels'] = jsonLabels

    # ic(jsonEntries)
    return rendered_im, labels_image, jsonEntries

if __name__ == "__main__":

    args = parse_command_line()

    # Should be started before creating any figure.
    if args.debug:
        # zvlog.start (('127.0.0.1', 7007))
        # zvlog.start ()
        zvApp = zv.App()
        zvApp.initialize()
        viewer = zvApp.getViewer()

    plt.ioff()

    if (args.no_antialiasing):
        mpl.rcParams['text.antialiased'] = False
        mpl.rcParams['lines.antialiased'] = False
        mpl.rcParams['patch.antialiased'] = False

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    process_next_image = True
    def rendered_image_callback(image_id, x, y, user_data):
        global process_next_image
        if zv.imgui.IsMouseClicked(zv.imgui.MouseButton.Left, False) and not zv.imgui.IsKeyDown(zv.imgui.Key.LeftCtrl):
            process_next_image = True

    for i in tqdm(range(args.num_images)):
        config = Config()
        config.scatter = args.scatter

        if viewer:
            process_next_image = False
        rendered, labels, jsonEntries = generate_plot (config, rendered_image_callback)

        prefix = str(args.output_dir / f"img-{i:05d}")

        # cv2 expects bgr
        cv2.imwrite(prefix + '.rendered.png', swap_rb(rendered))
        cv2.imwrite(prefix + '.labels.png', labels)
        with open(prefix + '.json', 'w') as f:
            f.write (json.dumps(jsonEntries))

        if viewer:
            while not process_next_image and zvApp.numViewers > 0:
                zvApp.updateOnce(1.0 / 30.0)

        # if i % 10 == 0:
        #     breakpoint()
        # time.sleep (0.5)

    if viewer:
        while zvApp.numViewers > 0:
            zvApp.updateOnce(1.0 / 30.0)

    # cv2.imshow ('Test Image', np.random.rand(128,128,3))
    # while True:
    #     cv2.waitKey (0)
