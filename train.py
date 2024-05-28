#!/usr/bin/env python3

# %% libs
from utils_main import *
from utils_load import *
from utils_tf import *

# %% new model settings
print("creating model: ", end='')
batch_size = 1000
num_epochs = 4
model, name = create_model()
model_name = f"all_{name}_batch{batch_size}_epoch{num_epochs}.keras"
print(model_name)

# %% load set
print("loading all datasets...")
images, labels, mapping = dataset_load_all()
mapping_n = len(mapping)

# %% preprocess set for tf
print("preparing images for tensorflow...")
train_input, train_target = prepdata(images, labels)
del images
gc.collect()
del labels
gc.collect()
print("images prepared.")

# %% training
print("training model...")
history = model.fit(train_input, train_target, batch_size=batch_size, epochs=num_epochs, verbose=2)
print("model trained.")

# %% save model
print(f"saving model: {model_name}")
model.save(d_models(model_name))
print("model saved.")

