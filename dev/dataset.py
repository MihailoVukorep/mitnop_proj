import os
import numpy as np
import gzip
import matplotlib.pyplot as plt

def read_emnist_labels(labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)
    return labels

def read_emnist_images(images_path: str, length: int):
    with gzip.open(images_path, 'rb') as imagesFile:
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        images = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16).reshape(length, 784).reshape(length, 28, 28, 1)
        images = images.transpose((0, 2, 1, 3)) 
    return images

def read_emnist_mapping(mapping_path: str):
    mapping = {}
    with open(mapping_path, 'r') as file:
        for line in file:
            words = line.strip().split()
            mapping[int(words[0])] = chr(int(words[1]))
    return mapping

def read_emnist_labels_mapped(labels_path: str, mapping_path: str):
    labels = read_emnist_labels(labels_path)
    # convert labels to actual mapping
    mapping = read_emnist_mapping(mapping_path)
    labels = np.array([mapping[element] for element in labels])
    return labels, mapping

def read_emnist(images_path: str, labels_path: str, mapping_path: str):
    labels, mapping = read_emnist_labels_mapped(labels_path, mapping_path)
    images = read_emnist_images(images_path, len(labels))
    return images, labels, mapping


def dataset_img(images, labels, index):
    plt.figure()
    plt.title(f'Example {index}. Label: {labels[index]}')
    plt.imshow(images[index].squeeze(), cmap=plt.cm.gray_r)
    plt.show()

def dataset_loadset(set_name, set_type):
    dataset_dir = "datasets/gzip/"
    path_images  = os.path.join(dataset_dir, f"emnist-{set_name}-{set_type}-images-idx3-ubyte.gz")
    path_labels  = os.path.join(dataset_dir, f"emnist-{set_name}-{set_type}-labels-idx1-ubyte.gz")
    path_mapping = os.path.join(dataset_dir, f"emnist-{set_name}-mapping.txt")
    return read_emnist(path_images, path_labels, path_mapping)


def dataset_load_all():
    return dataset_load(["train", "test"])

def dataset_load_train():
    return dataset_load(["train"])

def dataset_load_test():
    return dataset_load(["test"])

def dataset_load(types: list):
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    total_images = []
    total_labels = []
    total_mappings = {}
    for set_type in types:
        for set_name in names:
            images, labels, mapping = dataset_loadset(set_name, set_type)
            print(f'{set_type} {set_name} images: {images.shape}')
            print(f'{set_type} {set_name} labels: {labels.shape}')
            print(f'{set_type} {set_name} class: {len(mapping)}')
            total_images.append(images)
            total_labels.append(labels)
            total_mappings.update(mapping)

    total_images_nparr = None
    for arr in total_images:
        if total_images_nparr is None:
            total_images_nparr = arr
        else:
            total_images_nparr = np.concatenate([total_images_nparr, arr], axis=0)
    
    total_labels_nparr = None
    for arr in total_labels:
        if total_labels_nparr is None:
            total_labels_nparr = arr
        else:
            total_labels_nparr = np.concatenate([total_labels_nparr, arr], axis=0)

    return total_images_nparr, total_labels_nparr, total_mappings
            

def dataset_info():
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    types = ["train", "test"]
    print("printing info for all sets...")
    for set_type in types:
        for set_name in names:
            print(f"{set_name}-{set_type}:")
            dataset = {}
            dataset['images'], dataset['labels'], mapping = dataset_loadset(set_name, set_type)
            print('images:', dataset['images'].shape)
            print('labels:', dataset['labels'].shape)
            print('class:', len(mapping))
            print()

