#!/usr/bin/env python3

# %% libs

import math
import random as rand
import pandas as pd
import numpy as np
from utils_main import *
from utils_load import *

def charimgs_empty():
    charimgs = {chr(i): [] for i in range(48, 58)}        # '0'-'9'
    charimgs.update({chr(i): [] for i in range(97, 123)}) # 'a'-'z'
    charimgs.update({chr(i): [] for i in range(65, 91)})  # 'A'-'Z'
    return charimgs

def get_charimgs():
    images, labels, mapping = dataset_load_all()
    n = labels.shape[0]
    charimgs = charimgs_empty()

    for i in range(n):
        charimgs[labels[i]].append(images[i])

    return charimgs


def save_to_numpy(charimgs, prefix, output_dir='output'):
    images = []
    labels = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for k, v in charimgs.items():
        images.extend(v)
        labels.extend([k] * len(v))

    images = np.array(images)
    labels = np.array(labels)

    save_XXp(images, labels, prefix)


# %% sample

rand_seed = 47
print(f"using rand seed: {rand_seed}")
rand.seed(rand_seed)

charimgs = get_charimgs()
charimgs_80p = charimgs_empty()
charimgs_20p = charimgs_empty()

for k, v in charimgs.items():
    l = len(v)
    if l > 0:  # Ensure there are elements to sample from
        indices = list(range(l))
        rand.shuffle(indices)
        
        p = math.floor(0.8 * l)
        sampled_indices_80p = indices[:p]
        sampled_indices_20p = indices[p:]
        
        charimgs_80p[k].extend([v[i] for i in sampled_indices_80p])
        charimgs_20p[k].extend([v[i] for i in sampled_indices_20p])
        
        print(f"{k} : {l} --> 80% {p}, 20% {len(sampled_indices_20p)}")
    else:
        print(f"{k} : {l} --> no samples available")


save_to_numpy(charimgs_80p, '80p')
save_to_numpy(charimgs_20p, '20p')


# %%
