#!/usr/bin/env python3

import argparse, json, os, random, sys
import glob
from collections import namedtuple

import cv2
import numpy as np

patch_size = 23
half_patch_size = int(patch_size/2)
num_patches_per_image = 10

def parse_command_line():
    parser = argparse.ArgumentParser(description='Generating training images using OpenCV draw')
    parser.add_argument('input_dir', help='Input folder with all the images')
    parser.add_argument('output_dir', help='Output folder with all the patches')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_command_line()
    random.seed (42)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    patch_idx = 0
    for jsonFile in glob.iglob(args.input_dir + '/**/*.json', recursive=True):
        rendered_file = jsonFile.replace('.json', '.rendered.png')
        labels_file = jsonFile.replace('.json', '.labels.png')
        
        labels_image = cv2.imread(labels_file, 0)
        rendered_image = cv2.imread(rendered_file)
        with open(jsonFile, 'r') as f:
            input_json = json.load(f)
            rgb_color_by_label = {}
            for label_color in input_json:
                rgb_color_by_label[label_color['label']] = label_color['rgb_color']

        w, h  = labels_image.shape[1], labels_image.shape[1]
        for i in range(0, num_patches_per_image):
            label_should_be_background = random.randint(0, 1) == 1
            
            found = False
            patch_center = (-1, -1)
            label = None
            while not found:
                c = random.randint(half_patch_size, w-1-half_patch_size)
                r = random.randint(half_patch_size, h-1-half_patch_size)
                patch_center = (c,r)
                label = labels_image[r,c]
                found = ((label == 0 and label_should_be_background)
                         or (label > 0 and not label_should_be_background))

            print ('label', label)
            print ('patch_center', patch_center)
            rStart = patch_center[1] - half_patch_size
            rEnd = patch_center[1] + half_patch_size + 1
            cStart = patch_center[0] - half_patch_size
            cEnd = patch_center[0] + half_patch_size + 1
            patch_image = rendered_image[rStart:rEnd, cStart:cEnd, :]
            print ('patch_image', patch_image.shape)

            patch_prefix = os.path.join (args.output_dir, 'patch-{:09d}'.format(patch_idx))
            patch_image_path = patch_prefix + '.png'
            cv2.imwrite(patch_image_path, patch_image)
            patch_json_path = patch_prefix + '.json'
            patch_json = { 
                'rgb_color': rgb_color_by_label[label],
                'is_background': label_should_be_background,
            }
            with open(patch_json_path, 'w') as f:
                f.write (json.dumps(patch_json))

            patch_idx += 1
