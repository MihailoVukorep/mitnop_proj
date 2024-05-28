#!/usr/bin/env python3

import gzip
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os

from utils_main import *

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

def bytes_human_readable(num, suffix="B"):
    original = num
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix} ({original} bytes)"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix} ({original}) bytes"

def dataset_img(images, labels, index, note: str = ""):
    plt.figure()
    fullnote = ""
    if note:
        fullnote = f" -- {note}"
    plt.title(f'Example {index}. Label: {labels[index]}{fullnote}')
    plt.imshow(images[index].squeeze(), cmap=plt.cm.gray_r)
    plt.axis('off')
    plt.show()

def dataset_img2(images, labels, index, index2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
    ax1.set_title(f'Example {index}. Label: {labels[index]}')
    ax1.imshow(images[index].squeeze(), cmap=plt.cm.gray_r)
    ax1.axis('off')
    ax2.set_title(f'Example {index2}. Label: {labels[index2]}')
    ax2.imshow(images[index2].squeeze(), cmap=plt.cm.gray_r)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

def dataset_grid(images, startindex, grid_size):
    grid = images[startindex:startindex + grid_size*grid_size]
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        #ax.imshow(grid[i].reshape((28, 28)), cmap='gray')
        ax.imshow(grid[i].reshape((28, 28)), cmap=plt.cm.gray_r)
        ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


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

def images_to_md5s(images):
    hashes = []
    for image in images:
        arr_bytes = image.tobytes()
        md5_hash = hashlib.md5(arr_bytes).hexdigest()
        hashes.append(md5_hash)
    return hashes

def find_duplicates(lst):
    seen = {}
    dupe_x_y = list()
    duplicates = set()
    duplicates_i = set()
    for i, item in enumerate(lst):
        if item in seen:
            duplicates.add(item)
            duplicates_i.add(i)
            dupe_x_y.append((seen[item], i))
        else:
            seen[item] = i
    return duplicates_i, dupe_x_y


def dupes_check(images):
    hashes = images_to_md5s(images)
    return find_duplicates(hashes)

def dupes_check_all():
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    types = ["train", "test"]
    print("printing info for all sets...")
    for set_type in types:
        for set_name in names:
            print(f"{set_name}-{set_type}:")
            images, labels, mapping = dataset_loadset(set_name, set_type)
            print(f'images.: {images.shape}')
            print(f'labels.: {labels.shape}')
            print(f'class..: {len(mapping)}')
            print(f"bytes..: {bytes_human_readable(images.nbytes)}")
            
            duplicates_i, dupe_x_y = dupes_check(images) 
            print(f"dupes..: {len(duplicates_i)}")
            for i in dupe_x_y: dataset_img2(images, labels, i[0], i[1])

def dupes_rm(images, labels):
    duplicates_i, _ = dupes_check(images) 
    l = list(duplicates_i)
    images = np.delete(images, l, axis=0)
    labels = np.delete(labels, l, axis=0)
    return images, labels
    

def dataset_load(types: list):

    # check cache first
    s = '-'.join(types)
    cache_images_path = d_datasets(f"cache_{s}_images.npy")
    cache_labels_path = d_datasets(f"cache_{s}_labels.npy")
    cache_mapping_path = d_datasets(f"cache_{s}_mapping.npy")
    if os.path.exists(cache_images_path) and os.path.exists(cache_labels_path) and os.path.exists(cache_mapping_path):
        images = np.load(cache_images_path)
        labels = np.load(cache_labels_path)
        mapping_arr = np.load(cache_mapping_path)
        mapping = set(mapping_arr)
        print(f'loaded from cache images.: {images.shape}')
        print(f'loaded from cache labels.: {labels.shape}')
        print(f'loaded from cache class..: {len(mapping)}')
        print(f'loaded from cache bytes..: {bytes_human_readable(images.nbytes)}')
        return images, labels, mapping

    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    total_images = []
    total_labels = []
    total_mappings = {}
    for set_type in types:
        for set_name in names:
            images, labels, mapping = dataset_loadset(set_name, set_type)
            print(f'{set_type} {set_name} images.: {images.shape}')
            print(f'{set_type} {set_name} labels.: {labels.shape}')
            print(f'{set_type} {set_name} class..: {len(mapping)}')
            print(f'{set_type} {set_name} bytes..: {bytes_human_readable(images.nbytes)}')
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
    
    print(f'total images.: {total_images_nparr.shape}')
    print(f'total labels.: {total_labels_nparr.shape}')
    print(f'total class..: {len(total_mappings)}')
    print(f'total bytes..: {bytes_human_readable(total_images_nparr.nbytes)}')
    
    # remove dupes
    print("removing duplicates...")
    total_images_nparr, total_labels_nparr = dupes_rm(total_images_nparr, total_labels_nparr)
    print("duplicates removed.")
    print(f'unique images.: {total_images_nparr.shape}')
    print(f'unique bytes..: {bytes_human_readable(total_images_nparr.nbytes)}')

    # cache
    print("creating cache...")
    np.save(cache_images_path, total_images_nparr)
    np.save(cache_labels_path, total_labels_nparr)
    np.save(cache_mapping_path, np.array(list(total_mappings)))
    print("cache created")

    return total_images_nparr, total_labels_nparr, total_mappings

def dataset_info():
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    types = ["train", "test"]
    print("printing info for all sets...")
    for set_type in types:
        for set_name in names:
            print(f"{set_name}-{set_type}:")
            images, labels, mapping = dataset_loadset(set_name, set_type)
            print(f'images.: {images.shape}')
            print(f'labels.: {labels.shape}')
            print(f'class..: {len(mapping)}')
            print(f"bytes..: {bytes_human_readable(images.nbytes)}")
            print()




