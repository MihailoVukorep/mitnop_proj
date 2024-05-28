#!/usr/bin/env python3

# %% libs
from utils_main import *
from utils_load import *

# %% load
images, labels, mapping = dataset_load_all()

# %% DRAW GRID

dataset_grid(images, 0, 4).savefig(d_stats("grid1.png"), bbox_inches='tight', pad_inches=0)
dataset_grid(images, 4, 16).savefig(d_stats("grid2.png"), bbox_inches='tight', pad_inches=0)
dataset_grid(images, 16, 32).savefig(d_stats("grid3.png"), bbox_inches='tight', pad_inches=0)

# %%
