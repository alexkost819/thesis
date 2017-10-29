"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying RNN as a TPMS

"""
import functools
import logging
import os
import time
import tensorflow as tf

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'main_rnn.log'


def lazy_property(function):
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
    # Constants
    SIM_DATA_PATH = 'Data/simulation'
    SEQUENCE_LENGTH = int(1.75 / .001) + 1      # sec/sec
    EX_PER_EPOCH = 210
    CSV_N_COLUMNS = 5
    LABEL_UNDER = 0
    LABEL_NOM = 1
    LABEL_OVER = 2

    def __init__(self):
        """Constructor."""
        # VARYING ACROSS TESTS
        self.n_epochs = 250                 # number of times we go through all data
        self.n_hidden = 32                  # number of features per hidden layer in LSTM
        self.batch_size = 5                 # number of examples in a single batch
        self.n_layers = 3                   # number of hidden layers in model

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

        # CONSTANT
        self.n_features = 3                 # sprung_accel, unsprung_accel, sprung_height
        self.n_classes = 3                  # classifications: under, nominal, over pressure
        self.display_step = 10              # Every _ steps, save to tensorboard and display info
        self.shuffle = True                 # True if we want filenames to be shuffled, false if we don't
        self.logger = logging.getLogger(__name__)   # get the logger!

        # HYPERPARAMETERS TO BE DEFINED
        self.softmax_w = None
        self.softmax_b = None
        self.pred = None
        self.learning_rate = None
        self.optimizer = None

    def get_batches(self):
        with tf.name_scope("Get_Batches"):
            filenames = self._create_filename_list(self.SIM_DATA_PATH)
            filename_queue = self._create_filename_queue(filenames)

            # FEATURES AND LABELS
            # For dynamic_rnn, must have input be in certain shape
            # BEFORE: [batch_size x input_size x max_time]
            # AFTER:  [batch_size x max_time x input_size]
            # DIMS:   [depth x rows x columns]
            x, y = self._read_batch_from_queue(filename_queue)
            x = tf.transpose(x, [0, 2, 1])

        return x, y

    def prediction(self, inputs):
        with tf.name_scope("Model"):
            # add stacked layers if more than one layer
            if self.n_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([self._setup_lstm_cell() for _ in range(self.n_layers)],
                                                   state_is_tuple=True)
            else:
                cell = self._setup_lstm_cell()

            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

            # We transpose the output to switch batch size with sequence size.
            # http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
            outputs = tf.transpose(outputs, [1, 0, 2])

            # put the outputs into a classifier
            # weights = tf.Variable(shape=tf.random_normal([self.n_hidden, self.n_classes]), name='weights')
            # https://www.tensorflow.org/api_docs/python/tf/truncated_normal_initializer
            self.softmax_w = tf.get_variable('softmax_weights', [self.n_hidden, self.n_classes],
                                             initializer=tf.truncated_normal_initializer())
            self.softmax_b = tf.get_variable('softmax_biases', [self.n_classes])
            pred = tf.nn.xw_plus_b(outputs[-1], self.softmax_w, self.softmax_b)     # LOGITS
            # pred = tf.layers.dense(outputs[-1], n_classes)

            with tf.name_scope("Tensorboard"):
                tf.summary.histogram('weights', self.softmax_w)
                tf.summary.histogram('biases', self.softmax_b)
                tf.summary.histogram('pred', pred)

        return pred

    def optimize(self, pred, labels):
       with tf.name_scope("Optimize"):
            # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
            # http://colah.github.io/posts/2015-09-Visual-Information/
            # https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred)

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

    @staticmethod
    def evaluate(pred, labels):
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
                correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            with tf.name_scope('accuracy'):
                # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope("Tensorboard"):
                tf.summary.scalar('accuracy', accuracy)

        return accuracy

    def train_model(self):
        """ Step 0: Set up Tensorboard """
        timestamp = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "output"))
        run_dir = os.path.abspath(os.path.join(out_dir, "trained_model_" + timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with tf.name_scope("Tensorboard"):
            saver = tf.train.Saver(tf.global_variables())

        # Log the new run
        self._new_run_logging(timestamp)

        """ Step 1: Train the RNN """
        with tf.Session() as sess:
            # Initialization
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            writer = tf.summary.FileWriter(run_dir, sess.graph)
            summary_op = tf.summary.merge_all()
            step = 0  # should correspond with global_step

            self.logger.info("The training shall begin.")
            try:
                # while step < 2:
                # while True:
                while not coord.should_stop():
                    self.logger.debug('Step %d', step)

                    # Identify batches
                    batch_x, batch_y = sess.run([x, y])

                    # Train with batches defined above
                    sess.run(self.optimizer, feed_dict={
                        x: batch_x,
                        y: batch_y
                    })

                    """Step 3.2: Display training status and save model to TB """
                    if step % self.display_step == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=step)
                        acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
                        loss = cost.eval(feed_dict={x: batch_x, y: batch_y})
                        rate = learning_rate.eval()
                        iteration = step * self.batch_size
                        epoch = int(iteration / self.EX_PER_EPOCH)
                        self.logger.info('Epoch: {}, Iter: {}, Loss: {:.3f}, Accuracy: {:.3f}, Learning Rate: {:.3f}'\
                                         .format(epoch, iteration, loss, acc, rate))

                    summary = sess.run(summary_op)
                    writer.add_summary(summary, step)

                    step += 1

            except tf.errors.OutOfRangeError:
                self.logger.info('Cycled through epochs %d times', self.n_epochs)
            except KeyboardInterrupt:
                self.logger.info('Keyboard Interrupt? Gracefully quitting')
            finally:
                # Conclude training
                self.logger.info("The training is done.\n")
                coord.request_stop()
                coord.join(threads)
                writer.close()

    ''' Helper Functions '''
    @staticmethod
    def _split_data(data, val_size=0.2, test_size=0.2):
        """Spit all the data we have into training, validating, and test sets.

        By default, 64/16/20 split (20% of 80% = 16%)
        Credit: https://www.slideshare.net/TaegyunJeon1/electricity-price-forecasting-with-recurrent-neural-networks

        Args:
            data (list): Description
            val_size (float, optional): Percentage of data to be used for validation set
            test_size (float, optional): Percentage to validation set to be used for test set

        Returns:
            df_train (list): Description
            df_val (list): Description
            df_test (list): Description
        """
        ntest = int(round(len(data) * (1 - test_size)))
        nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

        df_train = data.iloc[:nval]
        df_val = data.iloc[nval:ntest]
        df_test = data.iloc[ntest:]

        return df_train, df_val, df_test

    @staticmethod
    def _create_filename_list(data_dir):
        """Identify the list of CSV files based on a given data_dir."""
        filenames = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".csv"):
                    rel_filepath = os.path.join(root, file)
                    abs_filepath = os.path.abspath(rel_filepath)
                    filenames.append(abs_filepath)

        return filenames

    def _create_filename_queue(self, filenames):
        """Ceate filename queue out of CSV files.

        Args:
            filenames (list of strings): list of filenames

        Returns:
            filename_queue: TF queue object
        """
        filename_queue = tf.train.string_input_producer(
            string_tensor=filenames,
            num_epochs=self.n_epochs,
            shuffle=self.shuffle)

        return filename_queue

    def _read_batch_from_queue(self, filename_queue):
        """Read CSV data and pack into tensors."""
        with tf.name_scope("Read_Batch"):
            # Step 1: Read CSV data
            reader = tf.TextLineReader(skip_header_lines=0,
                                       name='TextLineReader')

            features, labels = [], []

            # identify record_defaults used for decode_csv
            # default values and types if value is missing
            record_defaults = [[0.0] for _ in range(self.CSV_N_COLUMNS)]
            record_defaults[-1] = [0]

            for _ in range(self.batch_size):
                _, csv_row = reader.read_up_to(filename_queue, self.SEQUENCE_LENGTH)
                # content = [time, sprung_accel, unsprung_accel, sprung_height, label]
                content = tf.decode_csv(records=csv_row,
                                        record_defaults=record_defaults,
                                        name='decode_csv')

                # Parse content
                # content = [time, sprung_accel, unsprung_accel, sprung_height, label]
                normalized_columns = []
                for i in range(self.n_features):
                    raw_column = content[i + 1]
                    raw_max = tf.reduce_max(tf.abs(raw_column))
                    normalized_columns.append(tf.div(raw_column, raw_max))
                ex_features = tf.stack(normalized_columns)
                #ex_features = tf.stack(content[1:self.n_features+1])
                ex_labels = tf.one_hot(content[-1][0], self.n_classes)

                # Append each tensor to the list
                features.append(ex_features)
                labels.append(ex_labels)

            # Step 2: Stack lists of N-rank tensors to N+1 rank tensors
            features = tf.stack(features)
            labels = tf.stack(labels)

        return features, labels

        # BEFORE TRANSPOSE, Columns and Rows are reversed
        # BEFORE: [batch_size x input size] (vertical x horizontal dims)
        # we want it this: [batch_size, SEQUENCE_LENGTH, n_features])

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

    def _new_run_logging(self, timestamp):
        self.logger.info('*** NEW RUN ***')
        self.logger.info('filename: %s', "trained_model_" + timestamp)
        self.logger.info('n_epochs: %d', self.n_epochs)
        self.logger.info('n_hidden: %d', self.n_hidden)
        self.logger.info('batch_size: %d', self.batch_size)
        self.logger.info('n_layers: %d', self.n_layers)
        self.logger.info('exp_decay_enabled: %r', self.exp_decay_enabled)
        if self.exp_decay_enabled:
            self.logger.info('  exp_lr_starter_val: %f', self.exp_lr_starter_val)
            self.logger.info('  exp_lr_decay_steps: %f', self.exp_lr_decay_steps)
            self.logger.info('  exp_lr_decay_rate: %f', self.exp_lr_decay_rate)
        else:
            self.logger.info('static_lr_val: %f', self.static_lr_val)
        self.logger.info('Reg Type: %s', self.reg_type)
        self.logger.info('  Dropout Prob: %f', self.dropout_prob)
        self.logger.info('  Beta: %f\n', self.beta)


def main():
    """Sup Main."""

    # RUN 0: Identify Improvement of dropout, regularization, and # of Epochs
    # 0.0 - DROPOUT FALSE
    A = LSTMModel()
    A.train_model()
    # A.reset_model()

    # # 0.1 - DROPOUT TRUE
    # A.dropout_enabled = True
    # A.train_model()
    # A.reset_model()

    # # 0.2 - REGULARIZATION TRUE
    # A.reg_enabled = True
    # A.train_model()
    # A.reset_model()

    # # 0.3 - # of EPOCHS
    # A.n_epochs = 500
    # A.train_model()
    # A.reset_model()


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
