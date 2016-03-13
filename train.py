import helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Configuration
VALIDATION_SIZE = 2000
LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 1000
BATCH_SIZE = 100


# Set up data and convert to one hot
data = pd.read_csv('data/train.csv')
images = data.iloc[:,1:].values.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
labels = helpers.dense_to_one_hot(data[[0]].values.ravel(), 10).astype(np.uint8)


# Split data into cross validation and training set
cv_images = images[:VALIDATION_SIZE]
cv_labels = labels[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


# Initialize network variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])


# Use cross entropy cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# Train using stochastic gradient descent
for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = helpers.next_batch(train_images, train_labels, BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Calcluate and print accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: {}'.format(sess.run(accuracy, feed_dict={x: cv_images, y_: cv_labels})))
