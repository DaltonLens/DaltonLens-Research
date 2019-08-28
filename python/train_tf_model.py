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
    parser.add_argument('training_dir', help='Input folder with all the training patches')
    parser.add_argument('validation_dir', help='Input folder with all the validation patches')
    parser.add_argument('--model-name', help='Name of the model (output file prefix)', default="model")
    args = parser.parse_args()
    return args

def build_naive_dense_model():
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Flatten(name='flatten', input_shape=(patch_size,patch_size,3)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    # Add a softmax layer with 3 output units:
    model.add(layers.Dense(3, activation='linear', name='final'))
    return model

def build_naive_cnn_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(patch_size,patch_size,3)))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(patch_size,patch_size,3)))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='linear'))
    return model

def build_cnn_v2_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(patch_size,patch_size,3)))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='linear'))
    return model   

def build_cnn_v3_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, activation='linear', input_shape=(patch_size,patch_size,3)))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='linear'))
    return model

def split_rgb(rgb):
    def slice(i):
        def _slice(x):
            return tf.keras.backend.expand_dims(x[:,:,:,i])
        return _slice
    r = layers.Lambda(slice(0))(rgb)
    g = layers.Lambda(slice(1))(rgb)
    b = layers.Lambda(slice(2))(rgb)
    return [r,g,b]

def build_cnn_v4_model():
    # This returns a tensor    

    rgb = layers.Input(shape=(patch_size,patch_size,3))
    rgb_split = split_rgb(rgb)

    single_channel_input = layers.Input(shape=(patch_size,patch_size,1))
    x = layers.Conv2D(32, kernel_size=3, activation='linear')(single_channel_input)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    backbone_model = keras.Model(single_channel_input, x)

    # rgb_split = [layers.Lambda(tf.keras.backend.expand_dims(rgb[:, :, :, i])) for i in range(0, 3)]
    # rgb_split = [backbone_model(l) for l in rgb_split]
    # print (rgb_split)
    backbone_rgb = [backbone_model(c) for c in rgb_split]

    merged = keras.layers.concatenate(backbone_rgb)
    predictions = layers.Dense(3, activation='relu')(merged)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = keras.Model(rgb, predictions)
    return model

def build_cnn_v5_model():
    # This returns a tensor    

    rgb = layers.Input(shape=(patch_size,patch_size,3))
    rgb_split = split_rgb(rgb)

    single_channel_input = layers.Input(shape=(patch_size,patch_size,1))
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(single_channel_input)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = keras.layers.concatenate([x, layers.Flatten()(single_channel_input)])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    backbone_model = keras.Model(single_channel_input, x)

    # rgb_split = [layers.Lambda(tf.keras.backend.expand_dims(rgb[:, :, :, i])) for i in range(0, 3)]
    # rgb_split = [backbone_model(l) for l in rgb_split]
    # print (rgb_split)
    backbone_rgb = [backbone_model(c) for c in rgb_split]

    merged = keras.layers.concatenate(backbone_rgb)
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(3, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = keras.Model(rgb, predictions)
    return model

def build_cnn_v6_model():
    # This returns a tensor    

    rgb = layers.Input(shape=(patch_size,patch_size,3))
    rgb_split = split_rgb(rgb)

    single_channel_input = layers.Input(shape=(patch_size,patch_size,1))
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(single_channel_input)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = keras.layers.concatenate([x, layers.Flatten()(single_channel_input)])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    backbone_model = keras.Model(single_channel_input, x)

    # rgb_split = [layers.Lambda(tf.keras.backend.expand_dims(rgb[:, :, :, i])) for i in range(0, 3)]
    # rgb_split = [backbone_model(l) for l in rgb_split]
    # print (rgb_split)
    backbone_rgb = [backbone_model(c) for c in rgb_split]

    merged = keras.layers.concatenate(backbone_rgb)
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(3, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = keras.Model(rgb, predictions)
    return model

if __name__ == "__main__":
    args = parse_command_line()

    train_files = glob.glob("{}/*.tfrecords".format(args.training_dir))
    validation_files = glob.glob("{}/*.tfrecords".format(args.validation_dir))
    
    sess = tf.keras.backend.get_session()

    batch_size = 100

    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(lambda r: patch_example_parser(r))
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat()

    validation_dataset = tf.data.TFRecordDataset(validation_files)
    validation_dataset = validation_dataset.map(lambda r: patch_example_parser(r))
    validation_dataset = validation_dataset.shuffle(10000)
    validation_dataset = validation_dataset.batch(1)
    validation_dataset = validation_dataset.repeat()

    # epoch_steps = 20*10000*batch_size
    epoch_steps = 5000

    # model = build_naive_dense_model()
    # model = build_naive_cnn_model()
    # model = build_cnn_v2_model()
    # model = build_cnn_v4_model()
    # model = build_cnn_v5_model()
    model = build_cnn_v6_model()
    print (model.summary())

    # learning_rate = 0.001
    learning_rate = 0.002
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='mse',
                  metrics=[])

    cb_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

    save_path = args.model_name + '-{epoch:03d}-{val_loss:.3f}.hdf5'
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)

    # checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # # Create checkpoint callback
    # cb_save = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
    #                                              save_weights_only=True,
    #                                              verbose=1)

    model.fit(train_dataset, epochs=500, steps_per_epoch=epoch_steps, callbacks=[cb_board, cb_checkpoint], validation_data=validation_dataset, validation_steps=1000)
    # model.fit(train_dataset, epochs=20, steps_per_epoch=200, callbacks=[cb_board])

    model.save(args.model_name + ".h5")
