#!/usr/bin/env python3

from utils_main import *

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model_v0():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(class_mapping_n, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["categorical_accuracy"])
    return (model, "v0")

def create_model_v1():
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
    return (model, "v1")

def create_model_v2():
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
    return (model, "v2")

def create_model_v3():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(class_mapping_n, activation="sigmoid"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["categorical_accuracy"])
    return (model, "v3")

def create_model():
    # select model here
    model, name = create_model_v3()
    #model.summary()
    return (model, name)

def prepdata(images, labels):
    target_labels = np.array([class_mapping[i] for i in labels])
    data_target = to_categorical(target_labels, class_mapping_n)
    data_input = images / 255
    return data_input, data_target

def datagen(images, labels):
    return ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)

