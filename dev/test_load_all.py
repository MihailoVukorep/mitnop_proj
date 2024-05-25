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
from sys import getsizeof

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical

from dataset import *

# %% map classes

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

# %% load model

model_path = "model.keras"
print(f"loading model: {model_path}")
model = load_model(model_path)
model.summary()
print("loaded model.")

# %% funcs

def test_on(test_images, test_labels, test_mapping): 
    #print("loaded datasets:")
    #print(f'test images: {test_images.shape}')
    #print(f'test labels: {test_labels.shape}')
    #print(f'test mapping: {len(test_mapping)}')
    #print()
    #print("preparing images for tensorflow...")
    test_target_labels = np.array([class_mapping[i] for i in test_labels])
    test_input = test_images / 255
    test_target = to_categorical(test_target_labels, class_mapping_n)
    #print("images prepared.")
    num_epochs = 3
    batch_size = 100
    #print("EVALUATE: ")
    results = model.evaluate(test_input, test_target, verbose=0)
    print(f"categorical_accuracy: {results[1]} - loss: {results[0]}")


# %% testing...

print("testing on all test datasets at once...")

images, labels, mapping = dataset_load_test()

print()
print(f'test images..: {images.shape}')
print(f'test labels..: {labels.shape}')
print(f'test mapping.: {len(mapping)}')
print(f"test bytes...: {bytes_human_readable(images.nbytes)}")
print()
print()
print("all-sets: ")
test_on(images, labels, mapping)
# %%
