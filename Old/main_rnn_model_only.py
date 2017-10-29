"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying RNN as a TPMS

"""
import functools
import logging
import tensorflow as tf

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'main_rnn.log'


def lazy_property(function):
    """Decorator to make life easier

    Credit: https://danijar.com/structuring-your-tensorflow-models/

    Args:
        function (TYPE): Description
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class LSTMModel(object):
    '''
    ALLRNNStuff is a class of all the methods we use in the original script
    TODO: Break this up between data manipulation and training the model
    '''
    def __init__(self, inputs, labels):
        """Constructor."""
        self.logger = logging.getLogger(__name__)   # get the logger!

        # VARYING ACROSS TESTS
        self.n_hidden = 32                  # number of features per hidden layer in LSTM
        self.n_layers = 3                   # number of hidden layers in model
        self.n_classes = 3                  # classifications: under, nominal, over pressure

        # LEARNING RATE
        # Exponential Decay Parameters
        # https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
        self.static_lr_val = 0.1
        self.exp_decay_enabled = False      # enable/disable exponentially decaying learning rate
        self.exp_lr_starter_val = 0.5
        self.exp_lr_decay_steps = 50
        self.exp_lr_decay_rate = 0.96

        # REGULARIZATION (TO AVOID OVERFITTING)
        self.reg_type = None                # 'L2 Loss' or 'Dropout'
        self.dropout_prob = 0.5             # dropout probability
        self.beta = 0.01                    # Regularization beta variable

        # HYPERPARAMETERS TO BE DEFINED
        self.inputs = inputs
        self.labels = labels
        self.softmax_w = None
        self.softmax_b = None
        self.pred = None
        self.optimizer = None
        self.learning_rate = None
        self.optimizer = None
        self.evaluate = None

    @lazy_property
    def prediction(self):
        with tf.name_scope("Model"):
            # add stacked layers if more than one layer
            if self.n_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([self._setup_lstm_cell() for _ in range(self.n_layers)],
                                                   state_is_tuple=True)
            else:
                cell = self._setup_lstm_cell()

            outputs, _ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32)

            # We transpose the output to switch batch size with sequence size.
            # http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
            outputs = tf.transpose(outputs, [1, 0, 2])

            # put the outputs into a classifier
            # weights = tf.Variable(shape=tf.random_normal([self.n_hidden, self.n_classes]), name='weights')
            # https://www.tensorflow.org/api_docs/python/tf/truncated_normal_initializer
            self.softmax_w = tf.get_variable('softmax_weights', [self.n_hidden, self.n_classes],
                                             initializer=tf.truncated_normal_initializer())
            self.softmax_b = tf.get_variable('softmax_biases', [self.n_classes])
            self.pred = tf.nn.xw_plus_b(outputs[-1], self.softmax_w, self.softmax_b)     # LOGITS
            # pred = tf.layers.dense(outputs[-1], n_classes)

            with tf.name_scope("Tensorboard"):
                tf.summary.histogram('weights', self.softmax_w)
                tf.summary.histogram('biases', self.softmax_b)
                tf.summary.histogram('pred', self.pred)

        return pred

    @lazy_property
    def optimize(self):
       with tf.name_scope("Optimize"):
            # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
            # http://colah.github.io/posts/2015-09-Visual-Information/
            # https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pred)

            # Reduce Mean: Computes the mean of elements across dimensions of a tensor.
            cost = tf.reduce_mean(cross_entropy, name='total')

            # http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
            # Loss function using L2 Regularization
            regularizer = None
            if self.reg_type == 'L2 Loss':
                regularizer = tf.nn.l2_loss(self.softmax_w)
                cost = tf.reduce_mean(cost + self.beta * regularizer)

            if self.exp_decay_enabled:
                global_step = tf.get_variable('global_step',
                                              shape=(),
                                              initializer=tf.zeros_initializer(),
                                              trainable=False)
                self.learning_rate = tf.train.exponential_decay(learning_rate=self.exp_lr_starter_val,
                                                                global_step=global_step,
                                                                decay_steps=self.exp_lr_decay_steps,
                                                                decay_rate=self.exp_lr_decay_rate,
                                                                staircase=True,
                                                                name='learning_rate')

                # Passing global_step to minimize() will increment it at each step.
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.optimizer.minimize(cost, global_step=global_step)

            else:
                self.learning_rate = tf.constant(self.static_lr_val)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

            with tf.name_scope("Tensorboard"):
                tf.summary.scalar('cross_entropy', cost)
                tf.summary.scalar('learning_rate', self.learning_rate)
                if regularizer is not None:
                    tf.summary.scalar('regularizer', regularizer)

    @lazy_property
    def evaluate(self):
        """Evaluate Model calculates the accuracy of the model

        Args:
            pred (TYPE): Description
            labels (TYPE): Description

        Returns:
            TYPE: Description
        """
        # tf.argmax = returns index of the highest entry in a tensor along some axis.
        # So here, tf.equal is comparing predicted label to actual label, returns list of bools
        with tf.name_scope("Evaluate"):
            with tf.name_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
            with tf.name_scope('accuracy'):
                # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope("Tensorboard"):
                tf.summary.scalar('accuracy', accuracy)

        return accuracy

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
        if self.reg_type == 'Dropout':
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_prob)

        return cell


def main():
    """Sup Main."""
    A = LSTMModel()
    A.prediction()
    A.optimize()
    A.evaluate()


if __name__ == '__main__':
    # create logger with 'spam_application'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOGFILE_NAME)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(DEFAULT_FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    main()
