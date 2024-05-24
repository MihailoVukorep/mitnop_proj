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
    seen = {}
    duplicates = set()
    for i, item in enumerate(lst):
        if item in seen:
            duplicates.add(item)
            print(f"Duplicate found at indices {seen[item]}, {i}")
        else:
            seen[item] = i
    return duplicates

duplicates = find_duplicates(hashes)

print(len(duplicates))

# %%

dataset_img(images, labels, 32371)
dataset_img(images, labels, 113009)

# %%
