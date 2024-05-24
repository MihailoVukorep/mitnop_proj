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
    df.to_csv(filename + ".txt")
    with open(filename + "_stat.txt", 'w') as file:
        file.write("sum     " + str(df['count'].sum()) + "\n")
        file.write(str(df['count'].describe()) + "\n")
    return df

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
train_counter, mappings = doload([settype])
train_df = save_counter(train_counter, f"dataset_count_{settype}")
save_mappings(mappings, f"mappings_{settype}.txt")

# %% loading test

settype = "test"
test_counter, mappings = doload([settype])
test_df = save_counter(test_counter, f"dataset_count_{settype}")
save_mappings(mappings, f"mappings_{settype}.txt")

# %% total counter

total_counter = train_counter + test_counter
total_df = save_counter(total_counter, f"dataset_count")


# %% plot counts

def plot_dfs(total_df, train_df, test_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.set_title("character count (sorted by count descending)")
    ax1.bar(total_df['character'], total_df['count'], color="blue", label="total")
    ax1.bar(train_df['character'], train_df['count'], color="green", label="train")
    ax1.bar(test_df['character'], test_df['count'], color="red", label="test")
    ax1.legend()
    total_df2 = total_df.sort_values(by='character', ascending=True)
    train_df2 = train_df.sort_values(by='character', ascending=True)
    test_df2 = test_df.sort_values(by='character', ascending=True)
    ax2.set_title("character count (sorted by character ascending)")
    ax2.bar(total_df2['character'], total_df2['count'], color="blue", label="total")
    ax2.bar(train_df2['character'], train_df2['count'], color="green", label="train")
    ax2.bar(test_df2['character'], test_df2['count'], color="red", label="test")
    ax2.legend()
    plt.savefig("character_count.png")
    plt.show()

plot_dfs(total_df, train_df, test_df)

# %%
