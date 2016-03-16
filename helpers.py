import numpy as np
import tensorflow as tf


epochs_completed = 0
index_in_epoch = 0

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def next_batch(images, labels, batch_size):

    global epochs_completed
    global index_in_epoch

    num_examples = images.shape[0]
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        # Finished epoch
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]

        # Start new epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return images[start:end], labels[start:end]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
