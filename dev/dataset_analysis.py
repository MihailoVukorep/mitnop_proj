#!/usr/bin/env python3

# %% libs
from utils import *

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

def counter_stat(load_func, set_type):
    images, labels, mapping = load_func()
    counter = labels_to_counter(labels)
    df = counter_to_df(counter, f"dataset_unqiue_count_{set_type}")
    return counter, df

# %% get counters

train_counter, train_df = counter_stat(dataset_load_train, "train")
test_counter, test_df = counter_stat(dataset_load_test, "test")
total_counter, total_df = counter_stat(dataset_load_all, "all")


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
    plt.savefig("dataset_unqiue_count_all.png")
    plt.show()

plot_dfs(total_df, train_df, test_df)

# %%
