#!/usr/bin/env python3

# %% libs
from utils_main import *
from utils_load import *
from utils_tf import *

# %% load set
print("loading all datasets...")
images, labels, mapping = dataset_load_all()
mapping_n = len(mapping)

# %% dataset info
print("using:")
print(f'images..: {images.shape}')
print(f'labels..: {labels.shape}')
print(f'mapping.: {mapping_n}')
print(f'bytes...: {bytes_human_readable(images.nbytes)}')
print()

# %% preprocess set for tf
print("preparing images for tensorflow...")
train_input, train_target = prepdata(images, labels)
print("images prepared.")


# %% create model
print("creating model:")
model = create_model()

# %% training basic model

num_epochs = 4
batch_size = 1000

print("training model...")
history = model.fit(
    train_input,
    train_target,
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=2
)
print("model trained.")

# %% save model
model_name = "model_all_bigbatch.keras"
print(f"saving model: {model_name}")
model.save(d_models(model_name))
print("model saved.")

