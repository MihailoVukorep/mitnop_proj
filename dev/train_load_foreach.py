#!/usr/bin/env python3

# %% libs
from utils import *

# %% model
model = create_model()

# %% training

print("training on all train datasets...")

names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
epochs = [6, 5, 4, 3, 5, 3]
batch_sizes = [1, 100, 100, 100, 100, 100]

for i, set_name in enumerate(names):
    print(f"+-[{i}]--- {set_name}-train:")
    train_images, train_labels, train_mapping = dataset_loadset(set_name, "train")

    print(f'| train images..: {train_images.shape}')
    print(f'| train labels..: {train_labels.shape}')
    print(f'| train mapping.: {len(train_mapping)}')
    print(f'| train bytes...: {bytes_human_readable(train_images.nbytes)}')

    # prep data
    train_input, train_target = prepdata(train_images, train_labels)

    # train
    num_epochs = epochs[i]
    batch_size = batch_sizes[i]
    print("training model...")
    cnn_results = model.fit(train_input, train_target, batch_size=batch_size, epochs=num_epochs, verbose=2)
    print("model trained.")

# %% save model
print("saving model...")
model.save(d_models('model_foreach.keras'))  # Save as HDF5 file
print("model saved.")
