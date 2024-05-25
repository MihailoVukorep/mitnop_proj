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

# plt.figure(figsize=(8, 6))
# plt.plot(range(1, num_epochs + 1), cnn_results.history["loss"], "o--", label="Training")
# plt.plot(range(1, num_epochs + 1), cnn_results.history["val_loss"], "o--", label="Validation")
# plt.ylim(0)
# plt.title("Performance of the neural network")
# plt.xticks(range(1, num_epochs + 1))
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend(title="Phase")
# plt.show()


# %% testing model

print("predicting...")
predicted_labels = np.argmax(model.predict(val_input, verbose=2), axis=1)
print("done.")


# %% find misses

i = -1
right = 0
wrong = 0
for a, b in zip(predicted_labels, test_labels):
    i += 1
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        wrong += 1
    else:
        right += 1

l = val_input.shape[0]
f = wrong/l
print(f"wrong predictions: {wrong}/{l} -- {wrong/l}")
print(f"right predictions: {right}/{l} -- {right/l}")

# %% see some misses

i = -1
for a, b in zip(predicted_labels, test_labels):
    i += 1
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        dataset_img(val_input, test_labels, i, note=f"predict: {a}")
        time.sleep(5)


# %% see correct

i = -1
for a, b in zip(predicted_labels, test_labels):
    i += 1
    if str(a) == str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        dataset_img(val_input, test_labels, i, note=f"predict: {a}")
        time.sleep(1)

# %%
