"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying RNN as a TPMS
"""

import logging
import os
import progressbar
import time
import tensorflow as tf

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'main_rnn.log'


class CNNModel(object):
    '''
    ALLRNNStuff is a class of all the methods we use in the original script
    TODO: Break this up between data manipulation and training the model
    '''
    # Constants
    SIM_DATA_PATH = 'Data/simulation'
    SEQUENCE_LENGTH = int((1.75-.45) / .001) + 1      # sec/sec
    CSV_N_COLUMNS = 5

    def __init__(self):
        """Constructor."""
        # VARYING ACROSS TESTS
        self.n_epochs = 20                   # number of times we go through all data
        self.n_hidden = 40                  # number of features per hidden layer in LSTM
        self.batch_size = 10                # number of examples in a single batch
        self.n_layers = 5                   # number of hidden layers in model

        # INPUT PIPELINE
        self.filenames = self.create_filename_list(self.SIM_DATA_PATH)
        self.ex_per_epoch = len(self.filenames)
        self.train_length = self.n_epochs * self.ex_per_epoch

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
        self.reg_type = 'Dropout'                # 'L2 Loss' or 'Dropout'
        self.dropout_prob = 0.5             # dropout probability
        self.beta = 0.01                    # Regularization beta variable

        # CONSTANT
        self.n_features = 3                 # sprung_accel, unsprung_accel, sprung_height
        self.n_classes = 3                  # classifications: under, nominal, over pressure
        self.shuffle = True                 # True if we want filenames to be shuffled, false if we don't
        self.logger = logging.getLogger(__name__)   # get the logger!
        self.normalize = False              # True = normalized features, false = raw

    @staticmethod
    def create_filename_list(data_dir):
        """Identify the list of CSV files based on a given data_dir."""
        filenames = []
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".csv"):
                    rel_filepath = os.path.join(root, filename)
                    abs_filepath = os.path.abspath(rel_filepath)
                    filenames.append(abs_filepath)

        return filenames

    @staticmethod
    def combine_csv_files(filenames):
        """Combine all CSVs into one big file."""
        output_filepath = os.path.abspath(os.path.join(SIM_DATA_PATH, 'combined.csv'))
        with open(output_filepath, 'a') as output_file:
            for filepath in filenames:
                with open(filepath, 'r') as read_file:
                    for line in read_file:
                        output_filepath.write(line)

    def create_filename_queue(self, filenames):
        """Ceate filename queue out of CSV files."""
        filename_queue = tf.train.string_input_producer(
            string_tensor=filenames,
            num_epochs=self.n_epochs,
            shuffle=self.shuffle)

        return filename_queue

    def read_batch_from_queue(self, filename_queue):
        """Read CSV data and pack into tensors."""
        with tf.name_scope("Read_Batch"):
            # Step 1: Read CSV data
            reader = tf.TextLineReader(name='TextLineReader')

            # identify record_defaults used for decode_csv
            # default values and types if value is missing
            record_defaults = [[0.0] for _ in range(self.CSV_N_COLUMNS)]
            record_defaults[0] = [0]

            # read lines
            _, value = reader.read(filename_queue, self.SEQUENCE_LENGTH)
            content = tf.decode_csv(records=value,
                                    record_defaults=record_defaults,
                                    name='decode_csv')

            label = content[0]
            time_series = tf.stack(content[1:])

            # shuffle the data to generate BATCH_SIZE sample pairs
            #   capacity = max num elements in the queue
            #   min_after_deqeue = min num elements in queue after dequeue;
            #                      used to ensure samples are sufficiently mixed
            data_batch, label_batch = tf.train.shuffle_batch([time_series, label],
                                                             batch_size=self.batch_size,
                                                             capacity=len(self.filenames),
                                                             min_after_dequeue=10*self.batch_size)

            return data_batch, label_batch

    @staticmethod
    def reset_model():
        """Reset the model to prepare for next run."""
        tf.reset_default_graph()

    def train_model(self):
        """Train the model."""
        with tf.name_scope("Input_Batch"):
            filename_queue = self.create_filename_queue(self.filenames)

            # FEATURES AND LABELS
            # For dynamic_rnn, must have input be in certain shape
            # BEFORE: [batch_size x input_size x max_time]
            # AFTER:  [batch_size x max_time x input_size]
            # DIMS:   [depth x rows x columns]
            x, y = self.read_batch_from_queue(filename_queue)
            x = tf.transpose(x, [0, 2, 1])

            # Usually, the first column contains the target labels
            data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
            data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')
            data_test, data_val = np.split(data_test_val, 2)
            # Usually, the first column contains the target labels
            X_train = data_train[:, 1:]
            X_val = data_val[:, 1:]
            X_test = data_test[:, 1:]|
            N = X_train.shape[0]            # number of examples
            Ntest = X_test.shape[0]
            D = X_train.shape[1]            # number of dimensions
            y_train = data_train[:, 0]      # all rows, column 0. so for us, all rows, column 5
            y_val = data_val[:, 0]
            y_test = data_test[:, 0]
            self.logger.info('We have %s observations with %s dimensions', N, D)
            # Organize the classes
            self.n_classes = len(np.unique(y_train))
            base = np.min(y_train)  #Check if data is 0-based
            if base != 0:
                y_train -= base
                y_val -= base
                y_test -= base

        """Hyperparameters"""
        num_filt_1 = 16         # Number of filters in first conv layer
        num_filt_2 = 14         # Number of filters in second conv layer
        num_fc_1 = 40           # Number of neurons in fully connected layer
        self.dropout_prob = 1.0  # Dropout rate in the fully connected layer
        initializer = tf.contrib.layers.xavier_initializer()
        bn_train = tf.placeholder(tf.bool)          # Boolean value to guide batchnorm
                                                    # Set false when evaluating, set true when training

        # Define functions for initializing variables and standard layers
        # For now, this seems superfluous, but in extending the code
        # to many more layers, this will keep our code read-able

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

        with tf.name_scope("Reshaping_data"):
            x_image = tf.reshape(x, [-1, D, 1, 1])


        # CNN MODEL NOW
        """Build the graph"""
        # ewma is the decay for which we update the moving average of the
        # mean and variance in the batch-norm layers
        with tf.name_scope("Conv1"):
            W_conv1 = tf.get_variable("Conv_Layer_1",
                                      shape=[5, 1, 1, num_filt_1],
                                      initializer=initializer)
            b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
            a_conv1 = conv2d(x_image, W_conv1) + b_conv1

        with tf.name_scope('Batch_norm_conv1'):
            a_conv1 = tf.contrib.layers.batch_norm(a_conv1,
                                                   is_training=bn_train,
                                                   updates_collections=None)
            h_conv1 = tf.nn.relu(a_conv1)

        with tf.name_scope("Conv2"):
            W_conv2 = tf.get_variable("Conv_Layer_2",
                                      shape=[4, 1, num_filt_1, num_filt_2],
                                      initializer=initializer)
            b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
            a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

        with tf.name_scope('Batch_norm_conv2'):
            a_conv2 = tf.contrib.layers.batch_norm(a_conv2,
                                                   is_training=bn_train,
                                                   updates_collections=None)
            h_conv2 = tf.nn.relu(a_conv2)

        with tf.name_scope("Fully_Connected1"):
            W_fc1 = tf.get_variable("Fully_Connected_layer_1",
                                    shape=[D*num_filt_2, num_fc_1],
                                    initializer=initializer)
            b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
            h_conv3_flat = tf.reshape(h_conv2, [-1, D*num_filt_2])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        with tf.name_scope("Fully_Connected2"):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_prob)
            W_fc2 = tf.get_variable("W_fc2",
                                    shape=[num_fc_1, self.n_classes],
                                    initializer=initializer)
            b_fc2 = tf.Variable(tf.constant(0.1,
                                            shape=[self.n_classes]),
                                            name='b_fc2')
            h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.name_scope("Softmax"):
        #    regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
        #                  tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
        #                  tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
        #                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
        #                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))


            # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
            # http://colah.github.io/posts/2015-09-Visual-Information/
            # https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
            # Use sparse Softmax because we have mutually exclusive classes
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2, labels=y)

            # Reduce Mean: Computes the mean of elements across dimensions of a tensor.
            # https://www.tensorflow.org/api_docs/python/tf/reduce_mean
            cost = tf.reduce_mean(cross_entropy, name='total')

            # http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
            # Loss function using L2 Regularization
            regularizer = None
            if self.reg_type == 'L2 Loss':
                regularizer = tf.nn.l2_loss(weights)
                cost = tf.reduce_mean(cost + self.beta * regularizer)

        # EVALUATE OUR MODEL
        # tf.argmax = returns index of the highest entry in a tensor along some axis.
        # So here, tf.equal is comparing predicted label to actual label, returns list of bools
        with tf.name_scope("Evaluating"):
            with tf.name_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            with tf.name_scope('accuracy'):
                # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope("Optimizing"):
            if self.exp_decay_enabled:
                global_step = tf.get_variable('global_step', shape=(), initializer=tf.zeros_initializer(), trainable=False)
                learning_rate = tf.train.exponential_decay(learning_rate=self.exp_lr_starter_val,
                                                           global_step=global_step,
                                                           decay_steps=self.exp_lr_decay_steps,
                                                           decay_rate=self.exp_lr_decay_rate,
                                                           staircase=True,
                                                           name='learning_rate')

                # Passing global_step to minimize() will increment it at each step.
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   name='train').minimize(cost, global_step=global_step)

            else:
                learning_rate = tf.constant(self.static_lr_val)
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        """ Step 2: Set up Tensorboard """
        timestamp = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "output"))
        run_dir = os.path.abspath(os.path.join(out_dir, "trained_model_" + timestamp))
        # run_dir = os.path.abspath(os.path.join(out_dir, "trained_model"))
        checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with tf.name_scope("Tensorboard"):
            saver = tf.train.Saver(tf.global_variables())
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)
            tf.summary.histogram('pred', pred)
            tf.summary.scalar('cross_entropy', cost)
            if regularizer is not None:
                tf.summary.scalar('regularizer', regularizer)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('accuracy', accuracy)

        # Logging the Run
        self._new_run_logging(timestamp)

        """ Step 3: Train the RNN """
        with tf.Session() as sess:
            # Initialization
            bar = progressbar.ProgressBar(max_value=self.train_length)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            writer = tf.summary.FileWriter(run_dir, sess.graph)
            summary_op = tf.summary.merge_all()
            step = 0  # should correspond with global_step

            self.logger.info("The training shall begin.")
            try:
                bar.start()
                bar.update(0)
                while not coord.should_stop():     # only stops when n_epochs = 1...
                # while step <= self.train_length / self.batch_size:
                    # Train
                    _, acc, loss, rate, summary = sess.run([optimizer,
                                                            accuracy,
                                                            cost,
                                                            learning_rate,
                                                            summary_op])

                    # Display results every epoch
                    iteration = step * self.batch_size
                    if iteration % self.ex_per_epoch == 0:
                        saver.save(sess, checkpoint_prefix, global_step=step)
                        epoch = iteration / self.ex_per_epoch
                        self.logger.info('Epoch: {}, Loss: {:.3f}, Accuracy: {:.3f}, Learning Rate: {:.3f}'
                                         .format(epoch, loss, acc, rate))

                    writer.add_summary(summary, step)
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
    A = CNNModel()
    A.train_model()
    A.reset_model()


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
