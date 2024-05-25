#!/usr/bin/env python3

# %% libs
from utils import *

# %% load model

model_path = d_models("model.keras")
print(f"loading model: {model_path}")
model = load_model(model_path)
model.summary()
print("loaded model.")

# %% testing...

print("testing on all test datasets...")

names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
for set_name in names:
    print(f"{set_name}-test: ", end="")
    test_images, test_labels, test_mapping = dataset_loadset(set_name, "test")

    test_input, test_target = prepdata(test_images, test_labels)

    results = model.evaluate(test_input, test_target, verbose=0)
    print(f"categorical_accuracy: {results[1]} - loss: {results[0]}")

