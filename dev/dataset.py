import os
import numpy as np
import gzip
import matplotlib.pyplot as plt

def read_emnist(images_path: str, labels_path: str, mapping_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        images = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16).reshape(length, 784).reshape(length, 28, 28, 1)

    # convert labels to actual mapping
    mapping = {}
    with open(mapping_path, 'r') as file:
        for line in file:
            words = line.strip().split()
            mapping[int(words[0])] = chr(int(words[1]))
    labels = np.array([mapping[element] for element in labels])

    return images, labels, mapping


def dataset_img(images, labels, index):
    plt.figure()
    plt.title(f'Example {index}. Label: {labels[index]}')
    plt.imshow(images[index].squeeze(), cmap=plt.cm.gray_r)
    plt.show()

def dataset_load(set_name, set_type):
    dataset_dir = "datasets/gzip/"
    path_images  = os.path.join(dataset_dir, f"emnist-{set_name}-{set_type}-images-idx3-ubyte.gz")
    path_labels  = os.path.join(dataset_dir, f"emnist-{set_name}-{set_type}-labels-idx1-ubyte.gz")
    path_mapping = os.path.join(dataset_dir, f"emnist-{set_name}-mapping.txt")
    return read_emnist(path_images, path_labels, path_mapping)

def dataset_info():
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    types = ["train", "test"]
    print("printing info for all sets...")
    for set_type in types:
        for set_name in names:
            print(f"{set_name}-{set_type}:")
            dataset = {}
            dataset['images'], dataset['labels'], mapping = dataset_load(set_name, set_type)
            print('images:', dataset['images'].shape)
            print('labels:', dataset['labels'].shape)
            print('class:', len(mapping))
            print()

