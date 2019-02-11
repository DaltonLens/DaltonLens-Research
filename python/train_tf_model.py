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
    args = parser.parse_args()
    return args

def decode_image(image_bytes):
    image = tf.image.decode_png(image_bytes)
    image = tf.cast(image, tf.float32)
    image = tf.multiply(image, 1.0 / 255.)
    return image

def patch_example_parser(record):
    keys_to_features = {
        "image/is_background": tf.FixedLenFeature((), tf.int64, default_value=None),
        "image/rgb_color": tf.FixedLenFeature([3], tf.float32, default_value=None),
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=None),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=None),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=None),
    }

    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Perform additional preprocessing on the parsed data.
    image = decode_image(parsed["image/encoded"])    
    
    is_background = tf.cast(parsed["image/is_background"], tf.int32)
    rgb_color = parsed["image/rgb_color"]
    
    # features = {"image": image}
    
    return image, rgb_color

if __name__ == "__main__":
    args = parse_command_line()

    train_files = glob.glob("{}/*.tfrecords".format(args.input_dir))
    print (train_files)
    
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(lambda r: patch_example_parser(r))
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.batch(20)
    train_dataset = train_dataset.repeat()

    with tf.Session() as sess:
        it = train_dataset.make_one_shot_iterator()
        print (sess.run(it.get_next()))

    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Flatten(name='flatten', input_shape=(patch_size,patch_size,3)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 3 output units:
    model.add(layers.Dense(3, activation='relu', name='final'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['accuracy'])

    cb = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

    model.fit(train_dataset, epochs=20, steps_per_epoch=100, callbacks=[cb])