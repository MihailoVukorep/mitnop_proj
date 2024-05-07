
# %% libs
import random as rand
import os
import math
import cv2 as cv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical

# %% gen
rand.seed(89)

# %% loading data


directory = "Grapevine_Leaves_Image_Dataset"
species = ["Ak", "Ala_Idris", "Buzgulu", "Dimnit", "Nazli"]
species_mapping = {
    "Ak": 0,
    "Ala_Idris": 1,
    "Buzgulu": 2,
    "Dimnit": 3,
    "Nazli": 4,
}
species_cardinality = len(species)

dataset = []
labels = []

print("loading images:")

for specie in species:
    subdirectory = os.path.join(".", directory, specie)
    for file in os.listdir(subdirectory):
        path = os.path.join(subdirectory, file)
        image = cv.resize(cv.imread(path), (256, 256), interpolation=cv.INTER_AREA)
        dataset.append(image/255)
        labels.append(specie)

print("loaded images.")


# %% preping data

numeric_labels = [species_mapping[label] for label in labels]
categorical_labels = to_categorical(numeric_labels, species_cardinality)

# %% preping data for neural network training and validation

training_indices = rand.sample(range(len(dataset)), math.floor(0.8 * len(dataset)))
validation_indices = list(set(range(len(dataset))) - set(training_indices))

training_indices.sort()
validation_indices.sort()

training_set = np.array([dataset[i] for i in training_indices])
training_set_labels = categorical_labels[np.array(training_indices)]

validation_set = np.array([dataset[i] for i in validation_indices])
validation_set_labels = categorical_labels[np.array(validation_indices)]


# %% nn
cnn_model = Sequential()
cnn_model.add(Input(shape=(256,256,3)))
cnn_model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(2,2), padding="valid", activation="relu", use_bias=True))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(2,2), padding="valid", activation="relu", use_bias=True))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(species_cardinality, activation="softmax"))
cnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
cnn_model.summary()

# %% train nn
cnn_results = cnn_model.fit(training_set, training_set_labels, batch_size=32, epochs=20)

# %% nn validation
cnn_model.evaluate(validation_set, validation_set_labels, verbose=2)

# %%
