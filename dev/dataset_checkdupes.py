#!/usr/bin/env python3


# %% libs
import os
import gzip
import numpy as np
import pandas as pd
import time
import collections
import hashlib
import matplotlib.pyplot as plt
from dataset import *

# %% load set

images, labels, mapping = dataset_load_all()

# %% set info
print(images.shape)
print(labels.shape)
print(len(mapping))


# %% calc hashes

print("calculating hashes for all images...")

hashes = []
for image in images:
    arr_bytes = image.tobytes()
    md5_hash = hashlib.md5(arr_bytes).hexdigest()
    hashes.append(md5_hash)

print("hashes calculated")

# %% find dupes

def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates

def find_duplicates_xy(lst):
    seen = {}
    dupe_x_y = list()
    duplicates = set()
    for i, item in enumerate(lst):
        if item in seen:
            duplicates.add(item)
            dupe_x_y.append((seen[item], i))
            #print(f"Duplicate found at indices {seen[item]}, {i}")
        else:
            seen[item] = i
    return duplicates, dupe_x_y

#duplicates = find_duplicates(hashes)
duplicates, xy = find_duplicates_xy(hashes)

print(len(duplicates))

# %% show dupes

for x, y in xy:
    dataset_img2(images, labels, x, y)
    time.sleep(1)


# %% show dupe by indexes

dataset_img2(images, labels, 32371, 0)

# %%
