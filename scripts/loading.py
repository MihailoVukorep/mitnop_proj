import tensorflow as tf
import tensorflow_datasets as tfds

# Load the EMNIST dataset
emnist_dataset = tfds.load('emnist', split='letters', shuffle_files=True)

# Split the dataset into training and testing sets
train_dataset = emnist_dataset['train']
test_dataset = emnist_dataset['test']

# Prepare the datasets
train_dataset = train_dataset.map(lambda example: (tf.cast(example['image'], tf.float32)/255.0, example['label']))
test_dataset = test_dataset.map(lambda example: (tf.cast(example['image'], tf.float32)/255.0, example['label']))

# Print the number of samples in each dataset
print('Number of training samples:', tf.data.experimental.cardinality(train_dataset).numpy())
print('Number of testing samples:', tf.data.experimental.cardinality(test_dataset).numpy())
