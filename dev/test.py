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

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical

from dataset import *

# %% load data for testing

test_images, test_labels, test_mapping = dataset_loadset("digits", "test")
test_mapping_n = len(test_mapping)

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

test_target_labels = np.array([class_mapping[i] for i in test_labels])


# %% preprocess set
val_input = test_images / 255
val_target = to_categorical(test_target_labels, test_mapping_n)


# %% load model

model_path = "model.keras"

print(f"loading model: {model_path}")
model = load_model(model_path)
model.summary()
print("loaded model.")

# %% eval model

print("EVALUATE: ")
results = model.evaluate(val_input, val_target, verbose=2)


# %% testing model

print("predicting...")
predicted_labels = np.argmax(model.predict(val_input, verbose=2), axis=1)
print("done.")


# %% find misses

i = -1
right = 0
wrong = 0
for a, b in zip(predicted_labels, test_labels):
    i += 1
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        wrong += 1
    else:
        right += 1

l = val_input.shape[0]
f = wrong/l
print(f"wrong predictions: {wrong}/{l} -- {wrong/l}")
print(f"right predictions: {right}/{l} -- {right/l}")

# %% see some misses

i = -1
for a, b in zip(predicted_labels, test_labels):
    i += 1
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        dataset_img(val_input, test_labels, i, note=f"predict: {a}")
        time.sleep(5)


# %% see correct

i = -1
for a, b in zip(predicted_labels, test_labels):
    i += 1
    if str(a) == str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        dataset_img(val_input, test_labels, i, note=f"predict: {a}")
        time.sleep(1)

# %%
