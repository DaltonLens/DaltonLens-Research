#!/usr/bin/env python3

import argparse, json, os, random, sys
from collections import namedtuple
from dlcharts.common.dataset import LabeledImage

from zv.log import zvlog
from tqdm import tqdm

import cv2
import numpy as np
from pathlib import Path
import time

num_classes_per_image = 4
threshold_extra_margin = 0

Line = namedtuple('Line', ['p1', 'p2', 'thickness'])
Circle = namedtuple('Circle', ['center', 'radius', 'thickness'])

def swap_rb(im):
    return np.ascontiguousarray(im[:,:,[2,1,0]])

args = None

def parse_command_line():
    parser = argparse.ArgumentParser(description='Generating training images using OpenCV draw')
    parser.add_argument('output_dir', help='Output folder for the generated images')
    parser.add_argument('--debug', help='Toggle visual debugging', default=False, action='store_const', const=True)
    parser.add_argument('--num-drawings', help='Number of distinct drawings', default=100, type=int)
    parser.add_argument('--num-images-per-drawing', help='Number of images with distinct colors per drawing', default=1, type=int)
    parser.add_argument('--background-dir', type=Path, help='Folder to pick natural images as background', default=None)
    parser.add_argument('--width', help='Image width', default=320, type=int)
    parser.add_argument('--height', help='Image height', default=240, type=int)
    args = parser.parse_args()
    return args

def random_bgr():
    return (random.randint(0,255), random.randint(0,255), random.randint (0,255))

def color_index_to_label(i):
    return i*64 + 32

class DrawingSet:
    def __init__(self, img_size):
        num_lines = random.randint(0, 2) if args.background_dir else random.randint(1, 6)
        num_circles = random.randint(0, 2) if args.background_dir else random.randint(1, 4)
        num_rects = random.randint(0, 2) if args.background_dir else 0
        self.image_buffer = np.zeros((img_size[1], img_size[0], 3), np.uint8)

        self.lines = []
        for i in range(0, num_lines):
            x1 = random.randint(0, img_size[0])
            x2 = random.randint(0, img_size[0])
            y1 = random.randint(0, img_size[1])
            y2 = random.randint(0, img_size[1])
            thickness = random.randint(1,3)
            self.lines.append( Line(p1=(x1,y1), p2=(x2,y2), thickness=thickness))

        self.rects = []
        for i in range(0, num_rects):
            x1 = random.randint(0, img_size[0])
            x2 = random.randint(0, img_size[0])
            y1 = random.randint(0, img_size[1])
            y2 = random.randint(0, img_size[1])
            thickness = random.randint(1,3)
            self.rects.append( Line(p1=(x1,y1), p2=(x2,y2), thickness=thickness))

        self.circles = []
        for i in range(0, num_circles):
            x = random.randint(0, img_size[0])
            y = random.randint(0, img_size[1])
            center = (x,y)
            radius = random.randint(5, img_size[0]//3 if args.background_dir else img_size[0]/2)
            thickness = random.randint(1,3)
            self.circles.append(Circle(center=center, radius=radius, thickness=thickness))

    def draw(self, image, color, aliased):
        for line in self.lines:
            lw = line.thickness if not aliased else max(line.thickness, 1)
            cv2.line(image, line.p1, line.p2, color, lw, cv2.LINE_AA)
        for rect in self.rects:
            lw = rect.thickness if not aliased else max(rect.thickness, 1)
            cv2.rectangle(image, rect.p1, rect.p2, color, lw, cv2.LINE_AA)
        for circle in self.circles:
            lw = circle.thickness if not aliased else max(circle.thickness, 1)
            cv2.circle(image, circle.center, circle.radius, color, lw, cv2.LINE_AA)

class Renderer:
    def __init__(self):
        self.drawing = DrawingSet((args.width, args.height))
        self.image_buffer_for_mask = np.zeros((args.height, args.width, 3), np.uint8)
    
    # Render on the image with the given color and update the mask with the given label
    def draw(self, image, mask, label, color):
        self.image_buffer_for_mask[:] = (255,255,255)
        self.drawing.draw (self.image_buffer_for_mask, (0,0,0), aliased=True)
        self.drawing.draw (image, color, aliased=False)
        _, mask2d = cv2.threshold(cv2.cvtColor(self.image_buffer_for_mask, cv2.COLOR_BGR2GRAY),254-threshold_extra_margin,255,cv2.THRESH_BINARY)
        mask2d = 255-mask2d
        # zvlog.image ('mask2d', mask2d)
        mask[mask2d > 0] = label

class TrainingImage:
    def __init__(self, image_buffer, label_image, colors_by_label):
        self.image_buffer = image_buffer
        self.label_image = label_image
        self.colors_by_label = colors_by_label

    def write (self, path_prefix):
        image_path = path_prefix + '.antialiased.png'
        labels_path = path_prefix + '.labels.png'
        json_path = path_prefix + '.json'

        cv2.imwrite (image_path, self.image_buffer)
        cv2.imwrite (labels_path, self.label_image)

        jsonEntries = {}
        jsonEntries['size_cols_rows'] = [self.image_buffer.shape[1], self.image_buffer.shape[0]]
        jsonEntries['tags'] = ['opencv', 'lines', 'uniform_background']
        jsonLabels = []
        for label, bgr_color in self.colors_by_label.items():
            jsonLabels.append({
                'label': label,
                'rgb_color': [bgr_color[2], bgr_color[1], bgr_color[0]],
            })
        jsonEntries['labels'] = jsonLabels

        with open(json_path, 'w') as f:
            f.write (json.dumps(jsonEntries))

        labeled_im = LabeledImage(Path(json_path))
        labeled_im.compute_labels_as_rgb ()
        cv2.imwrite(path_prefix + '.aliased.png', swap_rb(labeled_im.labels_as_rgb))

class ImageGenerator:
    def __init__(self, background_dir):
        self.background_images = []
        if background_dir:
            self.background_images = list(background_dir.glob('**/*.JPEG'))
        self.image_buffer = np.zeros((args.height, args.width, 3), np.uint8)
        self.mask_buffer = np.zeros((args.height, args.width, 1), np.uint8)
        self.renderers = []
        for i in range(0, num_classes_per_image):
            self.renderers.append (Renderer())    

    def generate_with_random_colors(self):
        colors = []
        bg_color = random_bgr()
        # bg_color = (255,255,255) # TEMP: white background.
        for i in range(0, num_classes_per_image):
            while True:
                candidate_bgr = random_bgr()
                good_candidate = True
                for other_bgr in colors + [bg_color]:
                    if (np.amax(np.abs(np.subtract(candidate_bgr, other_bgr))) < 20):
                        good_candidate = False
                        break
                if good_candidate:
                    colors.append(random_bgr())
                    break
        
        if self.background_images:
            bg_idx = random.randrange(0,len(self.background_images))
            height, width = self.image_buffer.shape[0], self.image_buffer.shape[1]
            self.image_buffer[:] = cv2.resize(cv2.imread(str(self.background_images[bg_idx]), cv2.IMREAD_COLOR), (width, height))
        else:
            self.image_buffer[:] = bg_color

        self.mask_buffer[:] = 0
        
        for i, renderer in enumerate(self.renderers):
            label = color_index_to_label(i)
            renderer.draw (self.image_buffer, self.mask_buffer, label, colors[i])

        colors_by_label = {}
        colors_by_label[0] = bg_color
        for i, color in enumerate(colors):
            colors_by_label[color_index_to_label(i)] = color

        training_image = TrainingImage(self.image_buffer, self.mask_buffer, colors_by_label)

        if args.debug:
            zvlog.image('rendered', swap_rb(self.image_buffer))
            zvlog.image('mask', self.mask_buffer)

        return training_image

if __name__ == "__main__":    
    args = parse_command_line()
    if args.debug:
        # zvlog.start (('127.0.0.1', 7007))
        zvlog.start ()
    random.seed (time.time())

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for generator_i in tqdm(range(0, args.num_drawings)):
        # print ('Generating {} / {}'.format(generator_i, args.num_drawings))
        image_generator = ImageGenerator(args.background_dir)
        for color_i in range(0, args.num_images_per_drawing):
            training_image = image_generator.generate_with_random_colors()
            training_image.write (os.path.join (args.output_dir, 'img-{:05d}-{:03d}'.format(generator_i, color_i)))

    if args.debug:
        cv2.waitKey()
