# %% libs
import os
import gzip
import numpy as np
import pandas as pd
import time
import collections
import matplotlib.pyplot as plt
from dataset import *

# %% 

def load(set_name, set_type):
    dataset_dir = "datasets/gzip/"
    path_labels  = os.path.join(dataset_dir, f"emnist-{set_name}-{set_type}-labels-idx1-ubyte.gz")
    path_mapping = os.path.join(dataset_dir, f"emnist-{set_name}-mapping.txt")
    return read_emnist_labels_mapped(path_labels, path_mapping)

def doload(types):
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    counter = collections.Counter()
    mappings = []
    for set_type in types:
        for set_name in names:
            labels, mapping = load(set_name, set_type)
            print(f'{set_name}-{set_type} labels: {labels.shape}')
            print(f'{set_name}-{set_type} mapping: {len(mapping)}')
            mappings.append(mapping)
            for i in labels:
                counter.update(i)
    return counter, mappings

def save_counter(counter, filename):
    df = pd.DataFrame.from_dict(counter, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['character', 'count']
    df.sort_values(by='count', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'index'
    df.to_csv(filename)
    return df

def counter_plot(df, title):
    plt.figure(figsize=(15,5))
    fulltitle1 = f"{title} -- character count (sorted by count descending)"
    plt.title(fulltitle1)
    plt.bar(df['character'], df['count'])
    plt.savefig(fulltitle1 + ".png")
    plt.show()
    df2 = df.sort_values(by='character', ascending=True)
    plt.figure(figsize=(15,5))
    fulltitle2 = f"{title} -- character count (sorted by character ascending)"
    plt.title(fulltitle2)
    plt.bar(df2['character'], df2['count'])
    plt.savefig(fulltitle2 + ".png")
    plt.show()

def counter_plot2(train_df, test_df):
    plt.figure(figsize=(15,5))
    fulltitle1 = "both -- character count (sorted by count descending)"
    plt.title(fulltitle1)
    plt.scatter(train_df['character'], train_df['count'], color="green", label="train")
    plt.scatter(test_df['character'], test_df['count'], color="red", label="test")
    plt.legend()
    plt.savefig(fulltitle1 + ".png")
    plt.show()

    train_df2 = train_df.sort_values(by='character', ascending=True)
    test_df2 = test_df.sort_values(by='character', ascending=True)
    
    plt.figure(figsize=(15,5))
    fulltitle2 = "both -- character count (sorted by character ascending)"
    plt.title(fulltitle2)
    plt.scatter(train_df2['character'], train_df2['count'], color="green", label="train")
    plt.scatter(test_df2['character'], test_df2['count'], color="red", label="test")
    plt.legend()
    plt.savefig(fulltitle2 + ".png")
    plt.show()

def save_mappings(mappings, filename):
    with open(filename, 'w') as file:
        file.write("FULL MAPPINGS: \n")
        for mapping in mappings:
            file.write(str(mapping) + "\n")
        file.write("JUST VALUES: \n")
        for mapping in mappings:
            file.write(str(mapping.values()) + '\n')

# %% loading train

settype = "train"
counter, mappings = doload([settype])
train_df = save_counter(counter, f"dataset_count_{settype}.txt")
counter_plot(train_df, settype)
save_mappings(mappings, f"mappings_{settype}.txt")

# %% loading test

settype = "test"
counter, mappings = doload([settype])
test_df = save_counter(counter, f"dataset_count_{settype}.txt")
counter_plot(test_df, settype)
save_mappings(mappings, f"mappings_{settype}.txt")

# %% load both
counter, mappings = doload(["train", "test"])
df = save_counter(counter, f"dataset_count.txt")
counter_plot(df, "total")
save_mappings(mappings, f"mappings.txt")

# %% plot both

counter_plot2(train_df, test_df)

# %%
