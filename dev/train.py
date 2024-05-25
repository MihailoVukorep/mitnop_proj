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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical

from dataset import *

# %% load set

train_images, train_labels, train_mapping = dataset_load_train()
test_images, test_labels, test_mapping = dataset_load_test()

train_mapping_n = len(train_mapping)
test_mapping_n = len(test_mapping)

print("\n\n\n")

print(f'train images: {train_images.shape}')
print(f'train labels: {train_labels.shape}')
print(f'train mapping: {train_mapping_n}')

print(f'test images: {test_images.shape}')
print(f'test labels: {test_labels.shape}')
print(f'test mapping: {test_mapping_n}')

# %% make mapping for tensorflow

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

train_target_labels = np.array([class_mapping[i] for i in train_labels])
test_target_labels = np.array([class_mapping[i] for i in train_labels])


# %% preprocess set
train_input = train_images / 255
train_target = to_categorical(train_target_labels, train_mapping_n)

val_input = test_images / 255
val_target = to_categorical(test_target_labels, test_mapping_n)

# %% create model
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(2,2), padding="valid", activation="relu", use_bias=True))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu", use_bias=True))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(train_mapping_n, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
model.summary()

# %% training model params

num_epochs = 3
batch_size = 1

# %% train model

cnn_results = model.fit(train_input, train_target, validation_data=(val_input, val_target), batch_size=batch_size, epochs=num_epochs, verbose=2)

# %% plotting history

plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), cnn_results.history["loss"], "o--", label="Training")
plt.plot(range(1, num_epochs + 1), cnn_results.history["val_loss"], "o--", label="Validation")
plt.ylim(0)
plt.title("Performance of the neural network")
plt.xticks(range(1, num_epochs + 1))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(title="Phase")
plt.show()



# %% testing model

predicted_labels = np.argmax(model.predict(val_input, verbose=2), axis=1)
actual_labels = np.argmax(val_target, axis=1)

i = 0
for a, b in zip(predicted_labels, actual_labels):
    if a != b:
        print(f'{i} predict: {a} -- actual: {b}')
    i += 1


# %% individual images

index = 18
image = val_input[index:index+1]

plt.figure()
plt.imshow(image[0])
plt.show()

# %%
