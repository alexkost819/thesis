"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying RNN as a TPMS.

Attributes:
    DEFAULT_FORMAT (str): Logging format
    LOGFILE_NAME (str): Logging file name
    OUTPUT_DIR (str): Description
    SIM_LENGTH_FIX (int): bias to datapoint length due to slicing ops in Matlab, datapoints
    SIM_LENGTH_SEQ (int): simulation length, datapoints
    SIM_LENGTH_TIME (float): simulation time, sec
    SIM_RESOLUTION (float): simulation resolution, sec/datapoint
"""

# Basic Python
import logging
import tensorflow as tf

# Simulation Constants
OUTPUT_DIR = 'output'
SIM_LENGTH_TIME = 1.5 - .45
SIM_RESOLUTION = .001
SIM_LENGTH_FIX = 2
SIM_LENGTH_SEQ = int(SIM_LENGTH_TIME / SIM_RESOLUTION) + SIM_LENGTH_FIX


class LSTMModel(object):
    """
    LSTMModel is a class that builds and trains a RNN Model with LSTM cells.
    """

    def __init__(self, learning_rate, dropout_rate):
        """Constructor."""
        # VARYING ACROSS TESTS
        self.n_hidden = 30                          # number of features per hidden layer in LSTM
        self.n_layers = 3                           # number of hidden layers in model
        self.dropout_rate = dropout_rate            # dropout rate; 0.1 == 10% of input units drop out
        self.learning_rate = learning_rate          # learning rate, used for optimizing

        # CONSTANT
        self.n_features = 1                         # sprung_accel, unsprung_accel, sprung_height
        self.n_classes = 3                          # classifications: under, nominal, over pressure
        self.logger = logging.getLogger(__name__)   # get the logger!

        # MODEL PARAMETERS
        self.x = None
        self.y = None
        self.cost = None
        self.accuracy = None
        self.optimizer = None
        self.trainable = tf.placeholder(tf.bool, name='trainable')     # Guide batchnorm (false = evaluate, true=train)

        self.summary_op = None                      # summary operation to write data

    def build_model(self):
        """Build the RNN model."""
        input_shape = [None, SIM_LENGTH_SEQ, self.n_features] if self.n_features > 1 else [None, SIM_LENGTH_SEQ]
        self.x = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
        self.y = tf.placeholder(tf.int64, shape=[None], name='input_labels')

        with tf.variable_scope("Reshape_Data"):
            # tf.nn.conv2d requires inputs to be shaped as follows:
            # [batch_size, max_time, ...]
            # so -1 = batch size, should adapt accordingly
            # max_time = SIM_LENGTH_SEQ
            # ... = self.n_features
            x_reshaped = tf.reshape(self.x, [-1, SIM_LENGTH_SEQ, self.n_features])
            self.logger.debug('Input dims: {}'.format(x_reshaped.get_shape()))

        # add stacked layers if more than one layer
        if self.n_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([self._setup_lstm_cell() for _ in range(self.n_layers)],
                                               state_is_tuple=True)
        else:
            cell = self._setup_lstm_cell()

        # outputs = [batch_size, max_time, cell.output_size].
        outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                       inputs=x_reshaped,
                                       dtype=tf.float32)

        self.logger.debug('dynamic_rnn output dims: {}'.format(outputs.get_shape()))

        # We transpose the output to switch batch size with sequence size.
        # http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
        outputs = tf.transpose(outputs, [1, 0, 2])
        self.logger.debug('transpose output dims: {}'.format(outputs.get_shape()))

        with tf.variable_scope("Fully_Connected"):
            pred = tf.contrib.layers.fully_connected(inputs=outputs[-1],
                                                     num_outputs=self.n_classes,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     biases_initializer=tf.constant_initializer(0.1))
            self.logger.debug('FCon1 output dims: {}'.format(pred.get_shape()))
            tf.summary.histogram('pred', pred)

        # MEASURE MODEL ERROR
        # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
        #    http://colah.github.io/posts/2015-09-Visual-Information/
        #    https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
        #    Use sparse softmax because we have mutually exclusive classes
        # tf.reduce_mean = reduces tensor to mean scalar value of tensor
        with tf.variable_scope("Softmax"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
            self.cost = tf.reduce_mean(cross_entropy, name='total')
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

    """ Helper Functions """
    def _setup_lstm_cell(self):
        """Creates an LSTM Cell to be unrolled.

        There's a bug in tf.contrib.rnn.MultiRNNCell that requires we create
        new cells every time we want to a mult-layered RNN. So we use this
        helper function to create a LSTM cell with or without dropout.
        See more here: https://github.com/udacity/deep-learning/issues/132#issuecomment-325158949

        Returns:
            cell (BasicLSTMCell): BasicLSTM Cell with/without dropout
        """
        # forget_bias set to 1.0 b/c http://proceedings.mlr.press/v37/jozefowicz15.pdf
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)

        # add dropout
        if self.dropout_rate:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_rate)

        return cell
