"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying RNN as a TPMS

Attributes:
    DEFAULT_FORMAT (str): Logging format
    LOGFILE_NAME (str): Logging file name
"""

# Basic Python
import logging
import os
import progressbar
import time
import tensorflow as tf

# Alex Python
from DataProcessor import DataProcessor

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'main_rnn.log'


class LSTMModel(DataProcessor):
    '''
    ALLRNNStuff is a class of all the methods we use in the original script
    TODO: Break this up between data manipulation and training the model
    '''
    # Constants
    SIM_DATA_PATH = 'Data/simulation_orig'
    OUTPUT_DIR = 'output'
    SEQUENCE_LENGTH = int(1.75 / .001) + 1      # sec/sec
    CSV_N_COLUMNS = 5

    def __init__(self):
        """Constructor."""
        # VARYING ACROSS TESTS
        self.n_epochs = 50                  # number of times we go through all data
        self.n_hidden = 30                  # number of features per hidden layer in LSTM
        self.batch_size = 32                # number of examples in a single batch
        self.n_layers = 3                   # number of hidden layers in model

        # LEARNING RATE
        # Exponential Decay Parameters
        # https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
        self.static_lr_val = 0.05
        self.exp_decay_enabled = False      # enable/disable exponentially decaying learning rate
        self.exp_lr_starter_val = 0.5
        self.exp_lr_decay_steps = 50
        self.exp_lr_decay_rate = 0.96

        # REGULARIZATION TO AVOID OVERFITTING
        # http://uksim.info/isms2016/CD/data/0665a174.pdf - Use Dropout when n_hidden is large
        self.reg_type = 'Dropout'           # 'L2 Loss' or 'Dropout'
        self.dropout_prob = 0.5             # dropout probability
        self.beta = 0.01                    # Regularization beta variable

        # CONSTANT
        self.n_features = 3                 # sprung_accel, unsprung_accel, sprung_height
        self.n_classes = 3                  # classifications: under, nominal, over pressure
        self.shuffle = True                 # True if we want filenames to be shuffled, false if we don't
        self.logger = logging.getLogger(__name__)   # get the logger!
        self.normalize = False              # True = normalized features, false = raw

        # MODEL PARAMETERS
        self.optimizer = None
        self.accuracy = None
        self.cost = None
        self.learning_rate = None

        # INPUT PIPELINE
        super(LSTMModel, self).__init__(self.batch_size, self.n_epochs, self.n_classes, self.n_features)
        self.filenames = self._create_filename_list(self.SIM_DATA_PATH)
        self.use_all_files_for_training(self.filenames)

    def create_filename_queue(self, filenames):
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

    def read_batch_from_queue(self, filename_queue):
        """Read CSV data and pack into tensors."""
        with tf.name_scope("Read_Batch"):
            # Step 1: Read CSV data
            reader = tf.TextLineReader(skip_header_lines=0,
                                       name='TextLineReader')

            # identify record_defaults used for decode_csv
            # default values and types if value is missing
            record_defaults = [[0.0] for _ in range(self.CSV_N_COLUMNS)]
            record_defaults[-1] = [0]

            batch_features, batch_labels = [], []
            for _ in range(self.batch_size):
                _, csv_row = reader.read_up_to(filename_queue, self.SEQUENCE_LENGTH)
                # content = [time, sprung_accel, unsprung_accel, sprung_height, label]
                content = tf.decode_csv(records=csv_row,
                                        record_defaults=record_defaults,
                                        name='decode_csv')

                # Parse content
                ex_features = tf.stack(content[1:self.n_features+1])
                ex_labels = tf.one_hot(content[-1][0], self.n_classes)

                if self.normalize:
                    ex_features = tf.norm(ex_features, axis=0)

                # Append each tensor to the list
                batch_features.append(ex_features)
                batch_labels.append(ex_labels)

            # Step 2: Stack lists of N-rank tensors to N+1 rank tensors
            batch_features = tf.stack(batch_features)
            batch_labels = tf.stack(batch_labels)

        return batch_features, batch_labels

        # BEFORE TRANSPOSE, Columns and Rows are reversed
        # BEFORE: [batch_size x input size] (vertical x horizontal dims)
        # we want it this: [batch_size, SEQUENCE_LENGTH, n_features])

    @staticmethod
    def reset_model():
        """Reset the model to prepare for next run."""
        tf.reset_default_graph()

    def build_model(self):
        """Build the RNN model."""
        with tf.variable_scope("Input_Batch"):
            filename_queue = self.create_filename_queue(self.filenames)

            # FEATURES AND LABELS
            # For dynamic_rnn, must have input be in certain shape
            # BEFORE: [batch_size x input_size x max_time]
            # AFTER:  [batch_size x max_time x input_size]
            # DIMS:   [depth x rows x columns]
            x, y = self.read_batch_from_queue(filename_queue)
            x = tf.transpose(x, [0, 2, 1])

        # MODEL
        # add stacked layers if more than one layer
        if self.n_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([self._setup_lstm_cell() for _ in range(self.n_layers)],
                                               state_is_tuple=True)
        else:
            cell = self._setup_lstm_cell()

        sequence_lengths = []
        for _ in range(self.batch_size):
            sequence_lengths.append(self.SEQUENCE_LENGTH)

        outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                       inputs=x,
                                       sequence_length=sequence_lengths,
                                       dtype=tf.float32)

        # We transpose the output to switch batch size with sequence size.
        # http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
        outputs = tf.transpose(outputs, [1, 0, 2])

        with tf.variable_scope("Fully_Connected"):
            # put the outputs into a classifier
            # weights = tf.Variable(shape=tf.random_normal([self.n_hidden, self.n_classes]), name='weights')
            # https://www.tensorflow.org/api_docs/python/tf/truncated_normal_initializer
            weights = tf.get_variable('weights', [self.n_hidden, self.n_classes],
                                      initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', [self.n_classes])
            pred = tf.nn.xw_plus_b(outputs[-1], weights, biases)     # LOGITS
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)
            tf.summary.histogram('pred', pred)

        with tf.variable_scope("Softmax"):
            # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
            # http://colah.github.io/posts/2015-09-Visual-Information/
            # https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
            # Use sparse Softmax because we have mutually exclusive classes
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)

            # Reduce Mean: Computes the mean of elements across dimensions of a tensor.
            # https://www.tensorflow.org/api_docs/python/tf/reduce_mean
            self.cost = tf.reduce_mean(cross_entropy, name='total')
            tf.summary.scalar('cross_entropy_loss', self.cost)

            # http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
            # Loss function using L2 Regularization
            regularizer = None
            if self.reg_type == 'L2 Loss':
                regularizer = tf.nn.l2_loss(weights)
                self.cost = tf.reduce_mean(self.cost + self.beta * regularizer)

        # EVALUATE OUR MODEL
        # tf.argmax = returns index of the highest entry in a tensor along some axis.
        # So here, tf.equal is comparing predicted label to actual label, returns list of bools
        with tf.variable_scope("Evaluating"):
                correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope("Optimizing"):
            if self.exp_decay_enabled:
                global_step = tf.get_variable('global_step', shape=(), initializer=tf.zeros_initializer(), trainable=False)
                learning_rate = tf.train.exponential_decay(learning_rate=self.exp_lr_starter_val,
                                                           global_step=global_step,
                                                           decay_steps=self.exp_lr_decay_steps,
                                                           decay_rate=self.exp_lr_decay_rate,
                                                           staircase=True,
                                                           name='learning_rate')

                # Passing global_step to minimize() will increment it at each step.
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                        name='train').minimize(self.cost, global_step=global_step)

            else:
                learning_rate = tf.constant(self.static_lr_val)
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

            tf.summary.scalar('learning_rate', learning_rate)

    def train_model(self):
        """Train the model."""
        """ Step 2: Set up Tensorboard """
        timestamp = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, self.OUTPUT_DIR))
        run_dir = os.path.abspath(os.path.join(out_dir, "trained_model_" + timestamp))
        # run_dir = os.path.abspath(os.path.join(out_dir, "trained_model"))
        checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with tf.name_scope("Tensorboard"):
            saver = tf.train.Saver(tf.global_variables())

        # Logging the Run
        self._new_run_logging(timestamp)

        """ Step 3: Train the RNN """
        with tf.Session() as sess:
            # Initialization
            bar = progressbar.ProgressBar(max_value=self.train_length_ex)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            train_writer = tf.summary.FileWriter(run_dir + '/train', sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            train_writer = tf.summary.FileWriter(run_dir, sess.graph)
            summary_op = tf.summary.merge_all()
            step = 0  # should correspond with global_step

            self.logger.info("The training shall begin.")
            try:
                bar.start()
                bar.update(0)
                while not coord.should_stop():     # only stops when n_epochs = 1...
                # while step <= self.train_length_steps:
                    # Train
                    _, acc, loss, summary = sess.run([self.optimizer,
                                                      self.accuracy,
                                                      self.cost,
                                                      summary_op])

                    # Display results every epoch
                    iteration = step * self.batch_size
                    if iteration % self.ex_per_epoch == 0:
                        saver.save(sess, checkpoint_prefix, global_step=step)
                        epoch = iteration / self.ex_per_epoch
                        self.logger.info('Epoch: %d, Loss: %.3f, Accuracy: %.3f', epoch, loss, acc)

                    train_writer.add_summary(summary, step)
                    bar.update(iteration)
                    step += 1
            except tf.errors.OutOfRangeError:
                self.logger.info('Cycled through epochs %d times', self.n_epochs)
            except KeyboardInterrupt:
                self.logger.info('Keyboard Interrupt? Gracefully quitting')
            finally:
                step -= 1
                bar.finish()
                self.logger.info("The training is done.\n")
                coord.request_stop()
                coord.join(threads)
                train_writer.close()

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

    def _normalize_features(self, content):
        """Normalize Features normalizes each column one at a time.
        Can probably replace with tf.norm(axis=1), but not sure.d

        Args:
            content (TYPE): Description

        Returns:
            TYPE: Description
        """
        # http://www.faqs.org/faqs/ai-faq/neural-nets/part2/
        # See "Should I normalize/standardize/rescale the data?"
        normalized_columns = []
        for i in range(self.n_features):
            raw_column = content[i + 1]
            norm_column = tf.norm(raw_column)
        ex_features = tf.stack(norm_column)

        return ex_features

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
        self.logger.info('Normalization: %r', self.normalize)
        self.logger.info('exp_decay_enabled: %r', self.exp_decay_enabled)
        if self.exp_decay_enabled:
            self.logger.info('  exp_lr_starter_val: %.3f', self.exp_lr_starter_val)
            self.logger.info('  exp_lr_decay_steps: %.3f', self.exp_lr_decay_steps)
            self.logger.info('  exp_lr_decay_rate: %.3f', self.exp_lr_decay_rate)
        else:
            self.logger.info('static_lr_val: %.3f', self.static_lr_val)
        self.logger.info('Reg Type: %s', self.reg_type)
        self.logger.info('  Dropout Prob: %.2f', self.dropout_prob)
        self.logger.info('  Beta: %.3f\n', self.beta)


def main():
    """Sup Main."""
    A = LSTMModel()
    A.build_model()
    A.train_model()


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
