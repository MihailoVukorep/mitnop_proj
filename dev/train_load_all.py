#!/usr/bin/env python3

# %% libs
import os
import gzip
import numpy as np
import pandas as pd
import time
import cv2 as cv
import random as rand
import math
import matplotlib.pyplot as plt
import gc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical

from dataset import *

# %% load set

#train_images, train_labels, train_mapping = dataset_loadset("digits", "train")
#test_images, test_labels, test_mapping = dataset_loadset("digits", "test")

print("loading datasets...")

train_images, train_labels, train_mapping = dataset_load_train()
#test_images, test_labels, test_mapping = dataset_load_test()

train_mapping_n = len(train_mapping)
#test_mapping_n = len(test_mapping)

print("\n\n\n")

print("loaded datasets:")
print(f'train images..: {train_images.shape}')
print(f'train labels..: {train_labels.shape}')
print(f'train mapping.: {train_mapping_n}')
print(f'train bytes...: {bytes_human_readable(train_images.nbytes)}')
print()
#print(f'test images...: {test_images.shape}')
#print(f'test labels...: {test_labels.shape}')
#print(f'test mapping..: {test_mapping_n}')
#print(f'test bytes....: {bytes_human_readable(test_images.nbytes)}')
print()

# %% preprocess set for tf

print("preparing images for tensorflow...")

class_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
    'A': 36,
    'B': 37,
    'C': 38,
    'D': 39,
    'E': 40,
    'F': 41,
    'G': 42,
    'H': 43,
    'I': 44,
    'J': 45,
    'K': 46,
    'L': 47,
    'M': 48,
    'N': 49,
    'O': 50,
    'P': 51,
    'Q': 52,
    'R': 53,
    'S': 54,
    'T': 55,
    'U': 56,
    'V': 57,
    'W': 58,
    'X': 59,
    'Y': 60,
    'Z': 61
}
class_mapping_n = len(class_mapping)

train_target_labels = np.array([class_mapping[i] for i in train_labels])
del train_labels
gc.collect()
train_target = to_categorical(train_target_labels, class_mapping_n)
del train_target_labels
gc.collect()
train_input = train_images / 255
del train_images
gc.collect()

# test_target_labels = np.array([class_mapping[i] for i in test_labels])
# del test_labels
# gc.collect()
# test_target = to_categorical(test_target_labels, class_mapping_n)
# del test_target_labels
# gc.collect()
# test_input = test_images / 255
# del test_images
# gc.collect()

print("images prepared.")

# %% create model

print("creating model:")
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(2,2), padding="valid", activation="relu", use_bias=True))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu", use_bias=True))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(class_mapping_n, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
model.summary()

# %% training model params

num_epochs = 3
batch_size = 100

# %% train model

print("training model...")
#cnn_results = model.fit(train_input, train_target, validation_data=(val_input, val_target), batch_size=batch_size, epochs=num_epochs, verbose=2)
cnn_results = model.fit(train_input, train_target, batch_size=batch_size, epochs=num_epochs, verbose=2)
print("model trained.")

# %% save model

print("saving model...")
model.save('model.keras')  # Save as HDF5 file
print("model saved.")
