#!/usr/bin/env python3

import collections
import cv2 as cv
import gzip
import hashlib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import pandas as pd
import random as rand
import time
from sys import getsizeof
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

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


reversed_class_mapping = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'a',
    11: 'b',
    12: 'c',
    13: 'd',
    14: 'e',
    15: 'f',
    16: 'g',
    17: 'h',
    18: 'i',
    19: 'j',
    20: 'k',
    21: 'l',
    22: 'm',
    23: 'n',
    24: 'o',
    25: 'p',
    26: 'q',
    27: 'r',
    28: 's',
    29: 't',
    30: 'u',
    31: 'v',
    32: 'w',
    33: 'x',
    34: 'y',
    35: 'z',
    36: 'A',
    37: 'B',
    38: 'C',
    39: 'D',
    40: 'E',
    41: 'F',
    42: 'G',
    43: 'H',
    44: 'I',
    45: 'J',
    46: 'K',
    47: 'L',
    48: 'M',
    49: 'N',
    50: 'O',
    51: 'P',
    52: 'Q',
    53: 'R',
    54: 'S',
    55: 'T',
    56: 'U',
    57: 'V',
    58: 'W',
    59: 'X',
    60: 'Y',
    61: 'Z'
}

datasets_dir = "datasets/"
models_dir = "models/"

def d_datasets(file_path):
    return os.path.join(datasets_dir, file_path)

def d_models(model_name):
    return os.path.join(models_dir, model_name)

def read_emnist_labels(labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)
    return labels

def read_emnist_images(images_path: str, length: int):
    with gzip.open(images_path, 'rb') as imagesFile:
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        images = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16).reshape(length, 784).reshape(length, 28, 28, 1)
        images = images.transpose((0, 2, 1, 3)) 
    return images

def read_emnist_mapping(mapping_path: str):
    mapping = {}
    with open(mapping_path, 'r') as file:
        for line in file:
            words = line.strip().split()
            mapping[int(words[0])] = chr(int(words[1]))
    return mapping

def read_emnist_labels_mapped(labels_path: str, mapping_path: str):
    labels = read_emnist_labels(labels_path)
    # convert labels to actual mapping
    mapping = read_emnist_mapping(mapping_path)
    labels = np.array([mapping[element] for element in labels])
    return labels, mapping

def read_emnist(images_path: str, labels_path: str, mapping_path: str):
    labels, mapping = read_emnist_labels_mapped(labels_path, mapping_path)
    images = read_emnist_images(images_path, len(labels))
    return images, labels, mapping

def bytes_human_readable(num, suffix="B"):
    original = num
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix} ({original} bytes)"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix} ({original}) bytes"

def dataset_img(images, labels, index, note: str = ""):
    plt.figure()
    fullnote = ""
    if note:
        fullnote = f" -- {note}"
    plt.title(f'Example {index}. Label: {labels[index]}{fullnote}')
    plt.imshow(images[index].squeeze(), cmap=plt.cm.gray_r)
    plt.axis('off')
    plt.show()

def dataset_img2(images, labels, index, index2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
    ax1.set_title(f'Example {index}. Label: {labels[index]}')
    ax1.imshow(images[index].squeeze(), cmap=plt.cm.gray_r)
    ax1.axis('off')
    ax2.set_title(f'Example {index2}. Label: {labels[index2]}')
    ax2.imshow(images[index2].squeeze(), cmap=plt.cm.gray_r)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

def dataset_grid(images, grid_size):
    grid = images[:grid_size*grid_size]
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        #ax.imshow(grid[i].reshape((28, 28)), cmap='gray')
        ax.imshow(grid[i].reshape((28, 28)), cmap=plt.cm.gray_r)
        ax.axis('off')
    plt.show()


def dataset_loadset(set_name, set_type):
    dataset_dir = "datasets/gzip/"
    path_images  = os.path.join(dataset_dir, f"emnist-{set_name}-{set_type}-images-idx3-ubyte.gz")
    path_labels  = os.path.join(dataset_dir, f"emnist-{set_name}-{set_type}-labels-idx1-ubyte.gz")
    path_mapping = os.path.join(dataset_dir, f"emnist-{set_name}-mapping.txt")
    return read_emnist(path_images, path_labels, path_mapping)

def dataset_load_all():
    return dataset_load(["train", "test"])

def dataset_load_train():
    return dataset_load(["train"])

def dataset_load_test():
    return dataset_load(["test"])

def images_to_md5s(images):
    hashes = []
    for image in images:
        arr_bytes = image.tobytes()
        md5_hash = hashlib.md5(arr_bytes).hexdigest()
        hashes.append(md5_hash)
    return hashes

def find_duplicates(lst):
    seen = {}
    dupe_x_y = list()
    duplicates = set()
    duplicates_i = set()
    for i, item in enumerate(lst):
        if item in seen:
            duplicates.add(item)
            duplicates_i.add(i)
            dupe_x_y.append((seen[item], i))
        else:
            seen[item] = i
    return duplicates_i, dupe_x_y


def dupes_check(images):
    hashes = images_to_md5s(images)
    return find_duplicates(hashes)

def dupes_check_all():
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    types = ["train", "test"]
    print("printing info for all sets...")
    for set_type in types:
        for set_name in names:
            print(f"{set_name}-{set_type}:")
            images, labels, mapping = dataset_loadset(set_name, set_type)
            print(f'images.: {images.shape}')
            print(f'labels.: {labels.shape}')
            print(f'class..: {len(mapping)}')
            print(f"bytes..: {bytes_human_readable(images.nbytes)}")
            
            duplicates_i, dupe_x_y = dupes_check(images) 
            print(f"dupes..: {len(duplicates_i)}")
            for i in dupe_x_y: dataset_img2(images, labels, i[0], i[1])

def dupes_rm(images, labels):
    duplicates_i, _ = dupes_check(images) 
    l = list(duplicates_i)
    images = np.delete(images, l, axis=0)
    labels = np.delete(labels, l, axis=0)
    return images, labels
    

def dataset_load(types: list):
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    total_images = []
    total_labels = []
    total_mappings = {}
    for set_type in types:
        for set_name in names:
            images, labels, mapping = dataset_loadset(set_name, set_type)
            print(f'{set_type} {set_name} images.: {images.shape}')
            print(f'{set_type} {set_name} labels.: {labels.shape}')
            print(f'{set_type} {set_name} class..: {len(mapping)}')
            print(f'{set_type} {set_name} bytes..: {bytes_human_readable(images.nbytes)}')
            total_images.append(images)
            total_labels.append(labels)
            total_mappings.update(mapping)

    total_images_nparr = None
    for arr in total_images:
        if total_images_nparr is None:
            total_images_nparr = arr
        else:
            total_images_nparr = np.concatenate([total_images_nparr, arr], axis=0)
    
    total_labels_nparr = None
    for arr in total_labels:
        if total_labels_nparr is None:
            total_labels_nparr = arr
        else:
            total_labels_nparr = np.concatenate([total_labels_nparr, arr], axis=0)
    
    print(f'total images.: {total_images_nparr.shape}')
    print(f'total labels.: {total_labels_nparr.shape}')
    print(f'total class..: {len(total_mappings)}')
    print(f'total bytes..: {bytes_human_readable(total_images_nparr.nbytes)}')
    
    # remove dupes
    print("removing duplicates...")
    total_images_nparr, total_labels_nparr = dupes_rm(total_images_nparr, total_labels_nparr)
    print("duplicates removed.")
    print(f'unique images.: {total_images_nparr.shape}')
    print(f'unique bytes..: {bytes_human_readable(total_images_nparr.nbytes)}')

    return total_images_nparr, total_labels_nparr, total_mappings

def dataset_info():
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    types = ["train", "test"]
    print("printing info for all sets...")
    for set_type in types:
        for set_name in names:
            print(f"{set_name}-{set_type}:")
            images, labels, mapping = dataset_loadset(set_name, set_type)
            print(f'images.: {images.shape}')
            print(f'labels.: {labels.shape}')
            print(f'class..: {len(mapping)}')
            print(f"bytes..: {bytes_human_readable(images.nbytes)}")
            print()



def create_model():
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


def prepdata(images, labels):
    target_labels = np.array([class_mapping[i] for i in labels])
    del labels
    gc.collect()
    data_target = to_categorical(target_labels, class_mapping_n)
    del target_labels
    gc.collect()
    data_input = images / 255
    del images
    gc.collect()
    return data_input, data_target

