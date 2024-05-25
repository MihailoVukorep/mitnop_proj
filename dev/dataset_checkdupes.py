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

# %% check dupes all

dupes_check_all()

# %% load set
def unique(images, labels, mapping):
    print(f'images.: {images.shape}')
    print(f'labels.: {labels.shape}')
    print(f'class..: {len(mapping)}')
    print(f"bytes..: {bytes_human_readable(images.nbytes)}")

    print("removing duplicates...")
    images, labels = dupes_rm(images, labels)
    print("dupes removed.")

    print("unique images:")
    print(f'images.: {images.shape}')
    print(f"bytes..: {bytes_human_readable(images.nbytes)}")

# %% unique all
images, labels, mapping = dataset_load_all()
unique(images, labels, mapping)

# %% unique test
images, labels, mapping = dataset_load_train()
unique(images, labels, mapping)

# %% unique test
images, labels, mapping = dataset_load_test()
unique(images, labels, mapping)

# %%
