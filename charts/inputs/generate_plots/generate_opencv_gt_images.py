#!/usr/bin/env python3

import argparse, json, os, random, sys
from collections import namedtuple

import cv2
import numpy as np
import time

target_size = (256, 256)
num_classes_per_image = 4

Line = namedtuple('Line', ['p1', 'p2'])
Circle = namedtuple('Circle', ['center', 'radius'])

def parse_command_line():
    parser = argparse.ArgumentParser(description='Generating training images using OpenCV draw')
    parser.add_argument('output_dir', help='Output folder for the generated images')
    parser.add_argument('--debug', help='Toggle visual debugging', default=False, action='store_const', const=True)
    parser.add_argument('--num-drawings', help='Number of distinct drawings', default=100, type=int)
    parser.add_argument('--num-images-per-drawing', help='Number of images with distinct colors per drawing', default=100, type=int)
    args = parser.parse_args()
    return args

def random_bgr():
    return (random.randint(0,255), random.randint(0,255), random.randint (0,255))

def color_index_to_label(i):
    return i*64 + 32

class DrawingSet:
    def __init__(self, img_size):
        num_lines = 5
        num_circles = 3
        self.image_buffer = np.zeros((img_size[1], img_size[0], 3), np.uint8)

        self.lines = []
        for i in range(0, num_lines):
            x1 = random.randint(0, img_size[0])
            x2 = random.randint(0, img_size[0])
            y1 = random.randint(0, img_size[1])
            y2 = random.randint(0, img_size[1])
            self.lines.append( Line(p1=(x1,y1), p2=(x2,y2)) )

        self.circles = []
        for i in range(0, num_circles):
            x = random.randint(0, img_size[0])
            y = random.randint(0, img_size[1])
            center = (x,y)
            radius = random.randint(5, img_size[0]/2)
            self.circles.append(Circle(center=center, radius=radius))

    def draw(self, image, color):
        for line in self.lines:
            cv2.line(image, line.p1, line.p2, color, 1, cv2.LINE_AA)
        for circle in self.circles:
            cv2.circle(image, circle.center, circle.radius, color, 1, cv2.LINE_AA)

class Renderer:
    def __init__(self):
        self.drawing = DrawingSet(target_size)
        self.image_buffer_for_mask = np.zeros((target_size[1], target_size[0], 3), np.uint8)
    
    # Render on the image with the given color and update the mask with the given label
    def draw(self, image, mask, label, color):
        self.image_buffer_for_mask[:] = (255,255,255)
        self.drawing.draw (self.image_buffer_for_mask, (0,0,0))
        self.drawing.draw (image, color)
        _, mask2d = cv2.threshold(cv2.cvtColor(self.image_buffer_for_mask, cv2.COLOR_BGR2GRAY),254,255,cv2.THRESH_BINARY)
        mask2d = 255-mask2d
        cv2.imshow ('mask2d', mask2d)
        mask[mask2d > 0] = label

class TrainingImage:
    def __init__(self, image_buffer, label_image, colors_by_label):
        self.image_buffer = image_buffer
        self.label_image = label_image
        self.colors_by_label = colors_by_label

    def write (self, path_prefix):
        image_path = path_prefix + '.rendered.png'
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

class ImageGenerator:
    def __init__(self):
        self.image_buffer = np.zeros((target_size[1], target_size[0], 3), np.uint8)
        self.mask_buffer = np.zeros((target_size[1], target_size[0], 1), np.uint8)
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
            cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
            cv2.imshow('rendered', self.image_buffer)
            cv2.imshow('mask', self.mask_buffer)

        return training_image

if __name__ == "__main__":
    args = parse_command_line()
    random.seed (time.time())

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for generator_i in range(0, args.num_drawings):
        print ('Generating {} / {}'.format(generator_i, args.num_drawings))
        image_generator = ImageGenerator()
        for color_i in range(0, args.num_images_per_drawing):
            training_image = image_generator.generate_with_random_colors()
            training_image.write (os.path.join (args.output_dir, 'img-{:05d}-{:03d}'.format(generator_i, color_i)))

    if args.debug:
        cv2.waitKey()
