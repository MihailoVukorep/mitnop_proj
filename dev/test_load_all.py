#!/usr/bin/env python3

# %% libs
from utils import *


# %% load model

model_path = "model.keras"
print(f"loading model: {model_path}")
model = load_model(model_path)
model.summary()
print("loaded model.")


# %% testing...

print("loading test datasets...")
test_images, test_labels, test_mapping = dataset_load_test()

# %% set info

print()
print(f'test images..: {test_images.shape}')
print(f'test labels..: {test_labels.shape}')
print(f'test mapping.: {len(test_mapping)}')
print(f"test bytes...: {bytes_human_readable(test_images.nbytes)}")
print()
print("all-sets: ")

# %% prep data

test_input, test_target = prepdata(test_images, test_labels)

# %% eval

results = model.evaluate(test_input, test_target, verbose=2)

# %%
