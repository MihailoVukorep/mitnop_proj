#!/usr/bin/env python3


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

def labels_to_counter(labels):
    counter = collections.Counter()
    for i in labels:
        counter.update(i)
    return counter

def counter_to_df(counter, filename):
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

def counter_stat(set_type):
    labels, mappings = dataset_load_all_labels([set_type]) 
    counter = labels_to_counter(labels)
    df = counter_to_df(counter, f"dataset_count_{set_type}")
    return counter, df

# %% get counters

train_counter, train_df = counter_stat("train")
test_counter, test_df = counter_stat("test")
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
    plt.savefig("dataset_count.png")
    plt.show()

plot_dfs(total_df, train_df, test_df)

# %% set stat


def get_setcounts(types: list):
    names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    setcounts = []
    for set_type in types:
        for set_name in names:
            labels, mapping = dataset_loadlabels(set_name, set_type)
            setcounts.append((set_name, set_type, labels.shape[0]))
    return setcounts

def setcounts_to_df(data, filename):
    df = pd.DataFrame(data, columns=['set_name', 'set_type', 'count'])
    df.sort_values(by='count', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'index'
    df.to_csv(filename + ".txt")
    with open(filename + "_stat.txt", 'w') as file:
        file.write("sum     " + str(df['count'].sum()) + "\n")
        file.write(str(df['count'].describe()) + "\n")
    return df

def plot_setcounts(countsets):
    categories = list(set([item[0] for item in countsets]))
    data_types = list(set([item[1] for item in countsets]))
    counts = {category: {data_type: 0 for data_type in data_types} for category in categories}
    for category, data_type, count in countsets:
        counts[category][data_type] = count
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(categories))
    bar1 = [counts[category]['train'] for category in categories]
    bar2 = [counts[category]['test'] for category in categories]
    bars_train = ax.bar(index, bar1, bar_width, label='Train')
    bars_test = ax.bar([i + bar_width for i in index], bar2, bar_width, label='Test')
    ax.set_xlabel('Sets')
    ax.set_ylabel('Count')
    ax.set_title('Counts by setname')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(categories)
    ax.legend()
    plt.show()

def plot_setcounts2(countsets):
    categories = list(set([item[0] for item in countsets]))
    data_types = list(set([item[1] for item in countsets]))
    counts = {category: {data_type: 0 for data_type in data_types} for category in categories}
    for category, data_type, count in countsets:
        counts[category][data_type] = count
    index = range(len(categories))
    bar1 = [counts[category]['train'] for category in categories]
    bar2 = [counts[category]['test'] for category in categories]
    plt.bar(index, bar1, label='Train')
    plt.bar(index, bar2, label='Test')
    plt.xlabel('Sets')
    plt.xlabel('Count')
    plt.title('Counts by setname')
    plt.xticks(range(len(categories)), categories)
    plt.legend()
    plt.savefig("dataset_countbyset")
    plt.show()

# %% get set counts

setcounts = get_setcounts(["train", "test"])

# %% df
df = setcounts_to_df(setcounts, "dataset_countbyset")


# %% plot them
print(setcounts)
plot_setcounts2(setcounts)


# %%
