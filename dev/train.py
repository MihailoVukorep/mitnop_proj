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

train_images, train_labels, train_mapping = dataset_load("digits", "train")
test_images, test_labels, test_mapping = dataset_load("digits", "test")

train_mapping_n = len(train_mapping)
test_mapping_n = len(test_mapping)

print(f'train images: {train_images.shape}')
print(f'train labels: {train_labels.shape}')
print(f'train mapping: {train_mapping_n}')

print(f'test images: {test_images.shape}')
print(f'test labels: {test_labels.shape}')
print(f'test mapping: {test_mapping_n}')

# %% preprocess set
train_input = train_images / 255
train_target = to_categorical(train_labels, train_mapping_n)

val_input = test_images / 255
val_target = to_categorical(test_labels, test_mapping_n)

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
batch_size = 1000

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



# %%
