#!/usr/bin/env python3

# %% libs
import os
import gzip
import numpy as np
import pandas as pd
import time
from dataset import *

# %% print all dataset info 

images, labels, mapping = dataset_load_train()

# %% info

len(images)

# %%
