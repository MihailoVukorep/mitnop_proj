#!/usr/bin/env python3

# %% libs
from utils import *

# %% load set
print("loading datasets...")
images, labels, mapping = dataset_load_all()
mapping_n = len(mapping)

# %% dataset info
print("using datasets:")
print(f'train images..: {images.shape}')
print(f'train labels..: {labels.shape}')
print(f'train mapping.: {mapping_n}')
print(f'train bytes...: {bytes_human_readable(images.nbytes)}')
print()

# %% preprocess set for tf
print("preparing images for tensorflow...")
train_input, train_target = prepdata(images, labels)

print("images prepared.")

# %% create model
print("creating model:")
model = create_model()

# %% training model params
num_epochs = 4
batch_size = 100

# %% train model
print("training model...")
cnn_results = model.fit(train_input, train_target, batch_size=batch_size, epochs=num_epochs, verbose=2)
print("model trained.")

# %% save model
print("saving model...")
model.save('model_all.keras')
print("model saved.")

