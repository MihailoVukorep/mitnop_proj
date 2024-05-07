# %% libs

import random as rand
import os
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.math import confusion_matrix


# %% rng

rand.seed(89)


# %% Loading and organizing data

directory = "Grapevine_Leaves_Image_Dataset"
species = ["Ak", "Ala_Idris", "Buzgulu", "Dimnit", "Nazli"]
species_mapping_tb = {"Ak": 0, 
                      "Ala_Idris": 1, 
                      "Buzgulu": 2, 
                      "Dimnit": 3, 
                      "Nazli": 4}
species_mapping_bt = {0: "Ak", 
                      1: "Ala_Idris", 
                      2: "Buzgulu", 
                      3: "Dimnit", 
                      4: "Nazli"}
species_cardinality = len(species)

dataset = []
labels = []

for specie in species:
    subdirectory = os.path.join(".", directory, specie)
    for file in os.listdir(subdirectory):
        path = os.path.join(subdirectory, file)
        image = cv.resize(cv.imread(path, cv.IMREAD_GRAYSCALE), (256, 256), interpolation=cv.INTER_AREA)
        dataset.append(image / 255)
        labels.append(specie)

numeric_labels = [species_mapping_tb[label] for label in labels]
categorical_labels = to_categorical(numeric_labels, species_cardinality)


# %% Preparing data for training and validation

training_indices = rand.sample(range(len(dataset)), math.floor(0.8 * len(dataset)))
validation_indices = list(set(range(len(dataset))) - set(training_indices))

training_indices.sort()
validation_indices.sort()

training_set = np.array([dataset[i] for i in training_indices]) 
training_set_labels = categorical_labels[np.array(training_indices)]

validation_set = np.array([dataset[i] for i in validation_indices])
validation_set_labels = categorical_labels[np.array(validation_indices)]


# %% form model

batch_size = 1
num_epochs = 50

cnn_model = Sequential()
cnn_model.add(Input(shape=(256, 256, 1)))
cnn_model.add(Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu", use_bias=True))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding="valid", activation="relu", use_bias=True))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(filters=3, kernel_size=(2, 2), strides=(1, 1), padding="valid", activation="relu", use_bias=True))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(15, activation="relu"))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(species_cardinality, activation="softmax"))
cnn_model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["categorical_accuracy"])
cnn_model.summary()

# train model

cnn_results = cnn_model.fit(training_set, training_set_labels, batch_size=batch_size, epochs=num_epochs, validation_data=(validation_set, validation_set_labels))

# %% his

cnn_results.history


# %% graph

plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), cnn_results.history["categorical_accuracy"], "o--", label="Training")
plt.plot(range(1, num_epochs + 1), cnn_results.history["val_categorical_accuracy"], "o--", label="Validation")
plt.xlim(0)
plt.ylim(0)
plt.title("Performance of the neural network")
plt.xticks([i for i in range(1, num_epochs + 1) if i % 5 == 0])
plt.xlabel("Epoch")
plt.ylabel("Categorical Accuracy")
plt.legend(title="Phase")
plt.show()

predicted_validation_labels = np.argmax(cnn_model.predict(validation_set, verbose=0), axis=1)

true_validation_labels = np.array([numeric_labels[i] for i in validation_indices])

confusion_matrix_result = confusion_matrix(true_validation_labels, predicted_validation_labels)
print("Confusion matrix \n", confusion_matrix_result)

# %% print false

max_display = 5
display_count = 0

for index, true_label in enumerate(true_validation_labels):
    predicted_label = predicted_validation_labels[index]
    if true_label != predicted_label:
        if display_count < max_display:
            display_count += 1            
            cv.imshow(
                "({}) True={}, Predicted={}".format(
                    display_count, species_mapping_bt[true_label], 
                    species_mapping_bt[predicted_label]),
                validation_set[index]
            )
            cv.waitKey(3000)
        else:
            break

if display_count > 0:
    cv.destroyAllWindows()


# %%
