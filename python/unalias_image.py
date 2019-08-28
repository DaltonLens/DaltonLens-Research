#!/usr/bin/env python3

import argparse, json, os, random, sys
import glob
from collections import namedtuple

import cv2
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
layers = keras.layers

from common import *

def parse_command_line():
    parser = argparse.ArgumentParser(description='Train a tensorflow model to estimate the patch color')
    parser.add_argument('image', help='Input image')
    parser.add_argument('--model', help='Keras model to load', default='simple1.h5')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_command_line()

    model = tf.keras.models.load_model(args.model)
    print (model.summary())

    sess = tf.keras.backend.get_session()
    bgr_image = cv2.imread(args.image)
    # rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

    output_image = np.zeros(bgr_image.shape, np.uint8)

    for r in range(half_patch_size, bgr_image.shape[0]-half_patch_size):
        for c in range(half_patch_size, bgr_image.shape[1]-half_patch_size):
            rStart = r - half_patch_size
            rEnd = r + half_patch_size
            cStart = c - half_patch_size
            cEnd = c + half_patch_size
            patch = bgr_image[rStart:rEnd+1, cStart:cEnd+1, :]
            patch = patch.astype(np.float32)/255.0
            extended_patch = np.expand_dims(patch, axis=0)
            predicted_rgb = model.predict(extended_patch).reshape(3)
            predicted_rgb = (predicted_rgb * 255.0).astype(np.uint8)
            output_image[r,c,0] = predicted_rgb[2]
            output_image[r,c,1] = predicted_rgb[1]
            output_image[r,c,2] = predicted_rgb[0]
        print ('Row {} / {}'.format(r, bgr_image.shape[0]-half_patch_size))

    cv2.imshow('input', bgr_image)
    cv2.imshow('output', output_image)
    k = cv2.waitKey(0)
