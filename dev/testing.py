# %% libs
import os
import gzip
import numpy as np
import pandas as pd
from dataset import *

# %% print all dataset info 

dataset_info()

# %% 
print("loading dataset...")

dataset = {}
dataset['images'], dataset['labels'], mapping = dataset_load("digits", "train")

print("loaded set.")
print()
print("set info:")
print('images:', dataset['images'].shape)
print('labels:', dataset['labels'].shape)
print('class:', len(mapping))
print()

def img(index):
    dataset_img(dataset['images'], dataset["labels"], index)

# %% show images

for i in range(20, 30):
    img(i)

# %%
a = dataset['images'][0]
b = a.squeeze()
print()

img(0)

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