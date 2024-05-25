#!/usr/bin/env python3


# %% libs
from utils import *

# %% check dupes all

#dupes_check_all()

# %% load set
def unique(images, labels, mapping):
    print(f'images.: {images.shape}')
    print(f'labels.: {labels.shape}')
    print(f'class..: {len(mapping)}')
    print(f"bytes..: {bytes_human_readable(images.nbytes)}")

    print("removing duplicates...")
    images, labels = dupes_rm(images, labels)
    print("dupes removed.")

    print("unique images:")
    print(f'images.: {images.shape}')
    print(f"bytes..: {bytes_human_readable(images.nbytes)}")

    mapping_arr = np.array(list(mapping))
    np.save('datasets/cleanset_all_images.npy', images)
    np.save('datasets/cleanset_all_labels.npy', labels)
    np.save('datasets/cleanset_all_mapping.npy', mapping_arr)

# %% unique all
images, labels, mapping = dataset_load_all()
unique(images, labels, mapping)


