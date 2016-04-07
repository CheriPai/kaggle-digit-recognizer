import helpers as h
import numpy as np
import pandas as pd
import tensorflow as tf

# Set up data and convert to one hot
data = pd.read_csv('data/test.csv')
images = data.values.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)

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

W_fc1 = h.weight_variable([7 * 7 * 64, 1024])
b_fc1 = h.bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Create readout layer
W_fc2 = h.weight_variable([1024, 10])
b_fc2 = h.bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

sess = tf.Session()
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "data/output/model.ckpt")
    predict = tf.argmax(y_conv, 1)
    predictions = predict.eval(feed_dict={x: images})

    # Output results
    predictions = np.transpose(predictions)
    predictions = pd.DataFrame({'ImageId': np.arange(1, len(data.values) + 1),
                                'Label': predictions})
    predictions.to_csv('data/output/predictions.csv', index=False)
