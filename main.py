"""
Created on 24 June 2017

@author: Alex Kost

@description: Main python code file for Applying ANN as a TPMS

"""
#############
# Program Flow
#
# Things needed to do once:
#   Combine all CSVs to master CSVs, one for each pressure level
#   Save master csvs to a specific folder

# How should CSV be organized? https://www.tensorflow.org/programmers_guide/reading_data

# Things need to happen every time we run ANN
#   Import master CSVs
#   Count number of data points
#   Divide up to 50/25/25 for train/validate/test
#   Define each column to a tf.s
#

# second number is the tire pressure. We should try to merge all of the folders together into one big CSV
###############

from __future__ import print_function, division
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


""" Constants """
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def main():
    # Input parameters
    num_epochs = 100                # epoch is number of times we go through ALL data
    num_labels = 10                  # number of data labels: over-pressurized, under-pressurized, correct pressure
    length_of_time = 784           # how many datapoints in a time-series
    batch_size = 100                # See "Train our model"

    # INPUT LAYER
    # None = batch size can be any size
    x = tf.placeholder(tf.float32, [None, length_of_time])

    # WEIGHTS AND BIASES
    W = tf.Variable(tf.zeros([length_of_time, num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))

    # MORE ADVANCED: Weight initialization
    # Initialize weights with little noise for symmetry breaking and to prevent 0 gradients
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # MORE ADVANCED: Convolution and pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # stride is 1

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME') # stride is 1

    # CONVOLUTION LAYER
    # compute 32 features out of each 5x5 patch (dims 1/2 = patch size, dim 3 = inputs, dim 4 = output channels
    W_conv1 = weight_variable([5, 5, 1, 32])        # weights
    b_conv1 = bias_variable([32])                   # bias vector for each component
    x_image = tf.reshape(x, [-1, 28, 28, 1])        # -1 = inferred dimension, 28x28 images, number of color channels

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # CONVOLUTION LAYER #2
    # compute 64 features out of each 5x5 patch (dims 1/2 = patch size, dim 3 = inputs, dim 4 = output channels
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    # convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)             # The max_pool_2x2 method will reduce the image size to 14x14.

    # So now our 28x28 image is a 7x7... 5x5 twice: 28x28 --> 14x14 --> 7x7 (?)
    # oh because 32 features for each patch, then 64... maybe?

    # DENSELY CONVOLUTED LAYER
    W_fc1 = weight_variable([7 * 7 * 64, 1024])     #7 for each image length, 64 features, 1024 is arbitrary
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # DROPOUT
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # OUTPUT LAYER
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # TRAINING PARAMETERS
    # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth."
    # http://colah.github.io/posts/2015-09-Visual-Information/
    y_ = tf.placeholder(tf.float32, [None, num_labels])
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))      # inefficient
    cross_entropy = tf.reduce_mean(                                                             # efficient
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # use backpropogation algorithm (tweak this line for different optimizer)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # EVALUATE OUR MODEL
    # tf.argmax = returns index of the highest entry in a tensor along some axis.
    # So here, tf.equal is comparing predicted label to actual label, returns list of bools
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # TODO: mnist.test needs to be OURDATA.test

    # RUN SESSION
    #sess = tf.InteractiveSession()              # Launch model
    # better method is below
    # separates the creating the graph (model specification) and evaluating the graph (model fitting)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())             # initialize all tf.variables
        # Train our model
        for i in range(1000):                       # Run training step 1000 times
            # Each step of the loop, we get a "batch" of one hundred random data points from our training set.
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)        # 100 = batch size, not worth doing entire batch
            # TODO:  mnist.train needs to be OURDATA.train
            train_step.run(feed_dict={x: batch_xs,
                                      y_: batch_ys,
                                      keep_prob: 0.5
                                      })          # training with each batch

            # Print out how the training is doing every 100 steps
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_xs,
                    y_: batch_ys,
                    keep_prob: 1.0
                })
                print('step %d, training accuracy %g' % (i, train_accuracy))

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels,
            keep_prob: 1.0
        }))

if __name__ == '__main__':
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(DEFAULT_FORMAT)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    main()