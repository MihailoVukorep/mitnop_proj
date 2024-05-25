#!/usr/bin/env python3

# %% libs
from utils import *

# %% load data for testing

test_images, test_labels, test_mapping = dataset_loadset("digits", "test")
test_input, test_target = prepdata(test_images, test_labels)


# %% load model

model_path = d_models("model.keras")

print(f"loading model: {model_path}")
model = load_model(model_path)
model.summary()
print("loaded model.")

# %% eval model

print("EVALUATE: ")
results = model.evaluate(test_input, test_target, verbose=2)


# %% testing model

print("predicting...")
predicted_labels = np.argmax(model.predict(test_input, verbose=2), axis=1)
print("done.")

# %% find misses

a_lst = np.array([reversed_class_mapping[i] for i in predicted_labels])

right = 0
wrong = 0
for i, (a, b) in enumerate(zip(a_lst, test_labels)):
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        wrong += 1
    else:
        right += 1

l = test_labels.shape[0]
print(f"right predictions: {right}/{l} -- {right/l}")
print(f"wrong predictions: {wrong}/{l} -- {wrong/l}")

# %% see some misses

for i, (a, b) in enumerate(zip(a_lst, test_labels)):
    if str(a) != str(b):
        print(f'{i} predict: {a} -- actual: {b}')
        dataset_img(test_images, test_labels, i, note=f"predict: {a}")
        time.sleep(5)

# %%
