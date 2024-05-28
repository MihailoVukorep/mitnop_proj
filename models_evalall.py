#!/usr/bin/env python3

# %% libs
from utils_main import *
from utils_load import *
from utils_tf import *

images, labels, mapping = dataset_load_all()
test_input, test_target = prepdata(images, labels)
del images
gc.collect()
del labels
gc.collect()

for i, file in enumerate(os.listdir(models_dir)):
    print(f"model: {file} -- ", end='')
    full_path = os.path.join(models_dir, file)
    model = load_model(full_path)
    model.evaluate(test_input, test_target, verbose=2)
