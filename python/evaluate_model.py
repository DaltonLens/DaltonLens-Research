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
    parser.add_argument('input_dir', help='Input folder with all the patches')
    parser.add_argument('--model', help='Keras model to load', default='simple1.h5')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_command_line()

    train_files = glob.glob("{}/*.tfrecords".format(args.input_dir))
    print (train_files)
    
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(lambda r: patch_example_parser(r))
    
    model = tf.keras.models.load_model(args.model)
    print (model.summary())

    print ('Evaluation = ', model.evaluate (train_dataset.batch(30), steps=30))

    sess = tf.keras.backend.get_session()

    dataset_it = train_dataset.make_one_shot_iterator()
    # init = tf.global_variables_initializer()
    # sess.run (init)
    while True:
        raw_image, label = sess.run(dataset_it.get_next())
        image = raw_image * 255.0
        image = image.astype(np.uint8)
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        batch_raw_image = np.expand_dims(raw_image, axis=0)
        print (batch_raw_image.shape)
        predicted_rgb = model.predict(batch_raw_image).reshape(3)
        print (predicted_rgb.shape)
        predicted_rgb = (predicted_rgb * 255).astype(np.uint8)

        print ()
        print ('label = ', (label*255).astype(np.uint8))
        print ('imageCenter = ', rgb_image[half_patch_size, half_patch_size])
        print ('predicted = ', predicted_rgb)

        cv2.imshow('patch', image)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
