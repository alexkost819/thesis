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
    x = tf.placeholder(tf.float32, [None, length_of_time])

    # WEIGHTS AND BIASES
    W = tf.Variable(tf.zeros([length_of_time, num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))

    # OUTPUT LAYER
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # TRAINING PARAMETERS
    # Cross-Entropy: http://colah.github.io/posts/2015-09-Visual-Information/
    # "measuring how inefficient our predictions are for describing the truth."
    # inefficient method:
    y_ = tf.placeholder(tf.float32, [None, num_labels])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # efficient method:
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y)

    # use backpropogation algorithm (tweak this line for different optimizer)
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    # RUN SESSION
    sess = tf.InteractiveSession()              # Launch model
    tf.global_variables_initializer().run()     # initialize all tf.variables

    # Train our model
    for _ in range(1000):                       # Run training step 1000 times
        # Each step of the loop, we get a "batch" of one hundred random data points from our training set.
        # TODO:  mnist.train needs to be OURDATA.train
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)        # 100 = batch size, not worth doing entire batch
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # EVALUATE OUR MODEL
    # tf.argmax = returns index of the highest entry in a tensor along some axis.
    # So here, tf.equal is comparing predicted label to actual label, returns list of bools
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # TODO: mnist.test needs to be OURDATA.test
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



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