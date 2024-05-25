#!/usr/bin/env python3


# %% libs
from utils import *

# %% check dupes all

#dupes_check_all()

# %% load set
images, labels, mapping = dataset_load_all()
mapping_arr = np.array(list(mapping))
np.save(d_datasets('cleanset_all_images.npy'), images)
np.save(d_datasets('cleanset_all_labels.npy'), labels)
np.save(d_datasets('cleanset_all_mapping.npy'), mapping_arr)

# TODO: valid train test split