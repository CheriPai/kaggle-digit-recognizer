import helpers as h
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Configuration
VALIDATION_SIZE = 2000
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 20000
BATCH_SIZE = 50


# Set up data and convert to one hot
data = pd.read_csv('data/train.csv')
images = data.iloc[:,1:].values.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
labels = h.dense_to_one_hot(data[[0]].values.ravel(), 10).astype(np.uint8)


# Split data into cross validation and training set
cv_images = images[:VALIDATION_SIZE]
cv_labels = labels[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


# Initialize network variables
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])


# Create network layers
W_conv1 = h.weight_variable([5, 5, 1, 32])
b_conv1 = h.bias_variable([32])
h_conv1 = tf.nn.relu(h.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = h.max_pool_2x2(h_conv1)


W_conv2 = h.weight_variable([5, 5, 32, 64])
b_conv2 = h.bias_variable([64])
h_conv2 = tf.nn.relu(h.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = h.max_pool_2x2(h_conv2)


W_fc1 = h.weight_variable([7*7*64, 1024])
b_fc1 = h.bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Apply dropout to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Create readout layer
W_fc2 = h.weight_variable([1024, 10])
b_fc2 = h.bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])


# Use cross entropy cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()


# Train using stochastic gradient descent
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = h.next_batch(train_images, train_labels, BATCH_SIZE)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step {}, training accuracy {}".format(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})


print('Accuracy: {}'.format(sess.run(accuracy, feed_dict={x: cv_images, y_: cv_labels, keep_prob: 1.0})))
saver.save(sess, "data/output/model.ckpt")
