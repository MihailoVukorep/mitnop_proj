
# %% libs

import gzip
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# %% read func

def read_mnist(images_path: str, labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
                        .reshape(length, 784) \
                        .reshape(length, 28, 28, 1)

    return features, labels

# %% read data


train = {}
test = {}

# datasets/emnist-letters-train-images-idx3-ubyte.gz
# datasets/emnist-letters-train-labels-idx1-ubyte.gz

# datasets/emnist-letters-test-images-idx3-ubyte.gz
# datasets/emnist-letters-test-labels-idx1-ubyte.gz


train['features'], train['labels'] = read_mnist(
    'datasets/emnist-letters-train-images-idx3-ubyte.gz',
    'datasets/emnist-letters-train-labels-idx1-ubyte.gz'
)

test['features'], test['labels'] = read_mnist(
    'datasets/emnist-letters-test-images-idx3-ubyte.gz',
    'datasets/emnist-letters-test-labels-idx1-ubyte.gz'
)

print('training images:', train['features'].shape)
print('test images:', test['features'].shape)


# %% loaded data must have some pixels

for index, i in enumerate(test['features']):
    if not np.any(i):
        print(index)

# %% display image func

def display_image(position):
    image = train['features'][position].squeeze()
    plt.title('Example %d. Label: %d' % (position, train['labels'][position]))
    plt.imshow(image, cmap=plt.cm.gray_r)


# %% display

display_image(1000)

# TODO: fix labels mapping they are mapped by .txt file


# %%
