# %% libs
import os
import gzip
import numpy as np
import pandas as pd
import time
from dataset import *

# %% load
dataset = {}
dataset['images'], dataset['labels'], mapping = dataset_load("bymerge", "train")
print('images:', dataset['images'].shape)
print('labels:', dataset['labels'].shape)
print('class:', len(mapping))

# %% DRAW GRID

G = 32
grid = dataset['images'][:G*G]

# Plot the grid of images
fig, axes = plt.subplots(G, G, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    #ax.imshow(grid[i].reshape((28, 28)), cmap='gray')
    ax.imshow(grid[i].reshape((28, 28)), cmap=plt.cm.gray_r)
    ax.axis('off')

plt.show()

# %%
