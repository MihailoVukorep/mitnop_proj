#!/usr/bin/env python3

from utils_main import *

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def create_model_old():
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
    return model

def create_model():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(class_mapping_n, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["categorical_accuracy"])
    model.summary()
    return model

def prepdata(images, labels):
    target_labels = np.array([class_mapping[i] for i in labels])
    data_target = to_categorical(target_labels, class_mapping_n)
    data_input = images / 255
    return data_input, data_target

def datagen(images, labels):
    return ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)

