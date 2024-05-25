#!/usr/bin/env python3

# %% libs
import os
import gzip
import numpy as np
import pandas as pd
import time
from dataset import *

# %% load
images, labels, mapping = dataset_load_test()

# %% DRAW GRID

dataset_grid(images, 32)

# %%
