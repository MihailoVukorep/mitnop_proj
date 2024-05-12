# %% libs
import os
import gzip
import numpy as np
import pandas as pd
import time
from dataset import *

# %% print all dataset info 

dataset_info()

# %% 
print("loading dataset...")

dataset = {}
dataset['images'], dataset['labels'], mapping = dataset_load("bymerge", "train")

print("loaded set.")
print()
print("set info:")
print('images:', dataset['images'].shape)
print('labels:', dataset['labels'].shape)
print('class:', len(mapping))
print()

# %% img func

def img(index):
    dataset_img(dataset['images'], dataset["labels"], index)

# %% show images

for i in range(dataset['images'].shape[0]):
    img(i)
    time.sleep(0.5)

# %%
a = dataset['images'][0]
b = a.squeeze()
print()

img(0)

# %% DRAW GRID

G = 16  # Define the grid size
grid = dataset['images'][:G*G]  # Extract images for the grid, adjust indexing as needed

# Plot the grid of images
fig, axes = plt.subplots(G, G, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    #ax.imshow(grid[i].reshape((28, 28)), cmap='gray')
    ax.imshow(grid[i].reshape((28, 28)), cmap=plt.cm.gray_r)
    ax.axis('off')

plt.show()

# %% pixel sum enumerate

items = []
for index, item in enumerate(dataset['images']):
    sum_pixels = np.sum(item)
    items.append((index, sum_pixels))

# Step 3: Sort the list by pixel sum
df = pd.DataFrame(items, columns=['index', 'sum'])

# %% sorting

#df = df.sort_values(by=['sum'], ascending=True)

print(df.head())

df_min = df[df['sum'].min() == df['sum']]['index'].values[0]
df_max = df[df['sum'].max() == df['sum']]['index'].values[0]

# %%

setimg(df_min) # most pixels
setimg(df_max) # least pixels