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

if __name__ == "__main__":
    args = parse_command_line()

    train_files = glob.glob("{}/*.tfrecords".format(args.input_dir))
    print (train_files)
    
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(lambda r: patch_example_parser(r))
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.batch(30)
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
    model.add(layers.Dense(3, activation='linear', name='final'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='mse',
              metrics=['accuracy'])

    cb_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

    # checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # # Create checkpoint callback
    # cb_save = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
    #                                              save_weights_only=True,
    #                                              verbose=1)

    model.fit(train_dataset, epochs=20, steps_per_epoch=1000, callbacks=[cb_board])

    model.save("simple1.h5")
