#!/usr/bin/env python3

import argparse, json, os, random, sys
import glob
from collections import namedtuple

import cv2
import numpy as np

import tensorflow as tf

from common import *

num_patches_per_image = 10

def parse_command_line():
    parser = argparse.ArgumentParser(description='Generating training images using OpenCV draw')
    parser.add_argument('input_dir', help='Input folder with all the images')
    parser.add_argument('output_dir', help='Output folder with all the patches')
    args = parser.parse_args()
    return args

TrainingPatch = namedtuple('TrainingPatch', ['patch', 'rgb_color', 'is_background'])

class PatchGenerator:
    def __init__(self, args):
        self.args = args

    def oneshot_generator(self):
        patch_idx = 0
        for jsonFile in glob.iglob(self.args.input_dir + '/**/*.json', recursive=True):
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

                yield TrainingPatch(patch=patch_image, 
                                    rgb_color=rgb_color_by_label[label], 
                                    is_background=label_should_be_background)

                patch_idx += 1

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float32_feature(value):
    """Wrapper for inserting float32 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def patch_to_tfexample(image_data, image_format, height, width, rgb_color, is_background):

    rgb_color_as_float = [v/255.0 for v in rgb_color]

    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/rgb_color': float32_feature(rgb_color_as_float),
        'image/is_background': int64_feature(int(is_background)),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))

if __name__ == "__main__":
    args = parse_command_line()
    random.seed (42)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    patch_generator = PatchGenerator(args).oneshot_generator()

    tfrecords_filename = os.path.join (args.output_dir, 'patches.tfrecords')

    with tf.python_io.TFRecordWriter(tfrecords_filename) as tfrecord_writer, tf.Graph().as_default():        
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)
        
        with tf.Session('') as sess:
            for training_patch in patch_generator:
                png_string = sess.run(encoded_image,
                                    feed_dict={image_placeholder: training_patch.patch})

                example = patch_to_tfexample(png_string, b'png', 
                                             patch_size, patch_size, 
                                             training_patch.rgb_color,
                                             training_patch.is_background)
                tfrecord_writer.write(example.SerializeToString())
    
