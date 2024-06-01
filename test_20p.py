#!/usr/bin/env python3

# %% libs
from utils_main import *
from utils_load import *
from utils_tf import *
from utils_selectmodel import selectmodel
import time

# %% select model

model = selectmodel()

# %% libs
import pandas as pd
from utils_main import *
from utils_load import *
from utils_tf import *

# %% new model settings
print("loading...")
images, labels = load_XXp("20")

# %% preprocess set for tf
print("preparing images for tensorflow...")
test_input, test_target = prepdata(images, labels)
del images
gc.collect()
del labels
gc.collect()
print("images prepared.")

# %% eval model
print("EVALUATE: ")
results = model.evaluate(test_input, test_target, verbose=2)

# %%