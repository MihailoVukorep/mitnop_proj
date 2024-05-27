#!/usr/bin/env python3

# %% libs
from utils_main import *
from utils_load import *

# %% load
images, labels, mapping = dataset_load_test()

# %% DRAW GRID

dataset_grid(images, 32)

# %%
