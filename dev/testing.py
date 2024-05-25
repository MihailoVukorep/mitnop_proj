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