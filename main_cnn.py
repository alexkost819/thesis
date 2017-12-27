"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying CNN as a TPMS.
"""

# Basic Python
import logging

# Extended Python
import tensorflow as tf

# Alex Python
from data_processor import SIM_LENGTH_SEQ


class CNNModel(object):
    """
    CNNModel is a class that builds and trains a CNN Model.

    Attributes:
        accuracy (TensorFlow operation): step accuracy (predictions vs. labels)
        cost (TensorFlow operation): cross entropy loss
        dropout_rate (float): dropout rate; 0.1 == 10% of input units drop out
        learning_rate (float): learning rate, used for optimizing
        logger (logger object): logging object to write to stream/file
        n_classes (int): number of classifications: under, nominal, over pressure
        n_features (int): number of features in input feature data: sprung_accel
        num_fc_1 (int): number of neurons in first fully connected layer
        num_filt_1 (int): number of filters in first conv layer
        num_filt_2 (int): number of filters in second conv layer
        optimizer (TensorFlow operation): AdamOptimizer operation used to train the model
        summary_op (TensorFlow operation): summary operation of all tf.summary objects
        trainable (TensorFlow placeholder): boolean flag to separate training/evaluating
        x (TensorFlow placeholder): input feature data
        y (TensorFlow placeholder): input label data
    """

    def __init__(self, learning_rate, dropout_rate):
        """Constructor.

        Args:
            learning_rate (float): learning rate, used for optimizing
            dropout_rate (float): dropout rate; 0.1 == 10% of input units drop out
        """
        # HYPERPARAMETERS
        self.num_filt_1 = 16                        # number of filters in first conv layer
        self.num_filt_2 = 14                        # number of filters in second conv layer
        self.num_fc_1 = 40                          # number of neurons in first fully connected layer
        self.dropout_rate = dropout_rate            # dropout rate; 0.1 == 10% of input units drop out
        self.learning_rate = learning_rate          # learning rate, used for optimizing

        # CONSTANT
        self.n_features = 1                         # sprung_accel
        self.n_classes = 3                          # classifications: under, nominal, over pressure
        self.logger = logging.getLogger(__name__)   # get the logger!

        # MODEL MEMBER VARIABLES
        self.x = None                               # input data
        self.y = None                               # input label
        self.cost = None                            # cross entropy loss
        self.accuracy = None                        # step accuracy (predictions vs. labels)
        self.optimizer = None                       # optimizing operation
        self.trainable = tf.placeholder(tf.bool, name='trainable')  # flag to separate training/evaluating
        self.summary_op = None                      # summary operation to write data

    def build_model(self):
        """Build the CNN Model."""
        input_shape = [None, SIM_LENGTH_SEQ, self.n_features] if self.n_features > 1 else [None, SIM_LENGTH_SEQ]
        self.x = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
        self.y = tf.placeholder(tf.int64, shape=[None], name='input_labels')

        with tf.variable_scope("Reshape_Data"):
            # tf.nn.conv2d requires inputs to be shaped as follows:
            # [batch, in_height, in_width, in_channels]
            # so -1 = batch size, should adapt accordingly
            # in_height = "height" of the image (so one dimension)
            # in_width = "width" of image
            x_reshaped = tf.reshape(self.x, [-1, SIM_LENGTH_SEQ, 1, self.n_features])
            self.logger.debug('Input dims: {}'.format(x_reshaped.get_shape()))

        with tf.variable_scope("ConvBatch1"):
            conv1 = tf.contrib.layers.conv2d(inputs=x_reshaped,
                                             num_outputs=self.num_filt_1,
                                             kernel_size=[5, 1],
                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                             normalizer_params={'is_training': self.trainable,
                                                                'updates_collections': None})
            self.logger.debug('Conv1 output dims: {}'.format(conv1.get_shape()))

        with tf.variable_scope("ConvBatch2"):
            conv2 = tf.contrib.layers.conv2d(inputs=conv1,
                                             num_outputs=self.num_filt_2,
                                             kernel_size=[4, 1],
                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                             normalizer_params={'is_training': self.trainable,
                                                                'updates_collections': None})
            self.logger.debug('Conv2 output dims: {}'.format(conv2.get_shape()))

        with tf.variable_scope("Fully_Connected1"):
            conv2_flatten = tf.layers.flatten(conv2, name='Flatten')
            fc1 = tf.contrib.layers.fully_connected(inputs=conv2_flatten,
                                                    num_outputs=self.num_fc_1,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1),
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params={'is_training': self.trainable,
                                                                       'updates_collections': None})
            fc1 = tf.layers.dropout(inputs=fc1, rate=self.dropout_rate, training=self.trainable)
            self.logger.debug('FCon1 output dims: {}'.format(fc1.get_shape()))

        with tf.variable_scope("Fully_Connected2"):
            pred = tf.contrib.layers.fully_connected(inputs=fc1,
                                                     num_outputs=self.n_classes,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     biases_initializer=tf.constant_initializer(0.1))
            self.logger.debug('FCon2 output dims: {}'.format(pred.get_shape()))
            tf.summary.histogram('pred', pred)

        # MEASURE MODEL ERROR
        # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
        #    http://colah.github.io/posts/2015-09-Visual-Information/
        #    https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
        #    Use sparse softmax because we have mutually exclusive classes
        #    logits must be [batch_size, num_classes], label must be [batch_size]
        # tf.reduce_mean = reduces tensor to mean scalar value of tensor
        with tf.variable_scope("Softmax"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
            self.cost = tf.reduce_mean(cross_entropy, name='cost')
            tf.summary.scalar('cross_entropy_loss', self.cost)

        # EVALUATE OUR MODEL
        # tf.argmax = returns index of the highest entry in a tensor along some axis.
        #     Predictions are probabilities corresponding to class (ex. [0.7 0.2 0.1])
        #     tf.argmax returns the most probable label (ex. 0)
        # tf.equal = compares prediction to truth, returns list of bools (T if correct, F if not)
        # tf.reduce_mean = reduces tensor to mean scalar value of tensor
        # tf.cast = convert bools to 1 and 0
        with tf.variable_scope("Evaluating"):
            correct_pred = tf.equal(tf.argmax(pred, 1), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        # OPTIMIZE OUR MODEL
        with tf.variable_scope("Optimizing"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
