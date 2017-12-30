"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying RNN as a TPMS.
"""

# Basic Python
import logging

# Extended Python
import tensorflow as tf

# Alex Python
from data_processor import SIM_LENGTH_SEQ


class RNNModel(object):
    """
    RNNModel is a class that builds and trains a RNN model with LSTM cells.

    Attributes:
        accuracy (TensorFlow operation): step accuracy (predictions vs. labels)
        cost (TensorFlow operation): cross entropy loss
        dropout_rate (float): dropout rate; 0.1 == 10% of input units drop out
        learning_rate (float): learning rate, used for optimizing
        logger (logger object): logging object to write to stream/file
        n_classes (int): number of classifications: under, nominal, over pressure
        n_features (int): number of features in input feature data: sprung_accel
        n_hidden (int): number of features per hidden layer in RNN
        num_fc_1 (int): number of neurons in first fully connected layer
        n_layers (int): number of hidden layers in model
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
        self.n_hidden = 16                          # number of features per hidden layer in LSTM
        self.num_fc_1 = 40                          # number of neurons in first fully connected layer
        self.n_layers = 3                           # number of hidden layers in model
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
        """Build the RNN model."""
        input_shape = [None, SIM_LENGTH_SEQ, self.n_features] if self.n_features > 1 else [None, SIM_LENGTH_SEQ]
        self.x = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
        self.y = tf.placeholder(tf.int64, shape=[None], name='input_labels')

        if input_shape == [None, SIM_LENGTH_SEQ]:
            with tf.variable_scope("Reshape_Data"):
                # tf.nn.conv2d requires inputs to be shaped as follows:
                # [batch_size, max_time, ...]
                # so -1 = batch size, should adapt accordingly
                # max_time = SIM_LENGTH_SEQ
                # ... = self.n_features
                x_reshaped = tf.reshape(self.x, [-1, SIM_LENGTH_SEQ, self.n_features])
                self.logger.debug('Input dims: {}'.format(x_reshaped.get_shape()))

        with tf.variable_scope("LSTM_RNN"):
            # add stacked layers if more than one layer
            if self.n_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([self._setup_lstm_cell() for _ in range(self.n_layers)],
                                                   state_is_tuple=True)
            else:
                cell = self._setup_lstm_cell()

            # outputs = [batch_size, max_time, cell.output_size]
            #   outputs contains the output of the last layer for each time-step
            outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=x_reshaped,
                                           dtype=tf.float32)

            self.logger.debug('dynamic_rnn output dims: {}'.format(outputs.get_shape()))

            # We transpose the output to switch batch size with sequence size - http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
            outputs = tf.transpose(outputs, [1, 0, 2])      # Now shape = [max_time, batch_size, cell.output_size]
            last = outputs[-1]                              # Last slice is of shape [batch_size, cell.output_size]
            self.logger.debug('last output dims: {}'.format(last.get_shape()))

        with tf.variable_scope("Fully_Connected1"):
            fc1 = tf.contrib.layers.fully_connected(inputs=last,
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
        helper function to create a LSTM cell. See more here:
        https://github.com/udacity/deep-learning/issues/132#issuecomment-325158949

        Returns:
            cell (BasicLSTMCell): BasicLSTM Cell
        """
        # forget_bias set to 1.0 b/c http://proceedings.mlr.press/v37/jozefowicz15.pdf
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)

        return cell
