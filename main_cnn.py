"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying RNN as a TPMS
"""

import logging
import os
import progressbar
import time
import tensorflow as tf
import numpy as np

# Progressbar config
progressbar.streams.wrap_stderr()

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'main_cnn.log'


class CNNModel(object):
    '''
    CNNModel is a class that builds and trains a CNN Model
    '''
    # Constants
    SIM_DATA_PATH = 'Data/simulation_rnn'
    SEQUENCE_LENGTH = int((1.5 - .45) / .001) + 2   # sec/sec

    def __init__(self):
        """Constructor."""
        # VARYING ACROSS TESTS
        self.n_epochs = 50                          # number of times we go through all data
        self.batch_size = 10                        # number of examples in a single batch
        self.num_filt_1 = 16                        # number of filters in first conv layer
        self.num_filt_2 = 14                        # number of filters in second conv layer
        self.num_fc_1 = 40                          # number of neurons in fully connected layer

        # LEARNING RATE
        # Exponential Decay Parameters
        # https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
        self.static_lr_val = 0.05
        self.exp_decay_enabled = False              # enable/disable exponentially decaying learning rate
        self.exp_lr_starter_val = 0.5
        self.exp_lr_decay_steps = 50
        self.exp_lr_decay_rate = 0.96

        # REGULARIZATION TO AVOID OVERFITTING
        # http://uksim.info/isms2016/CD/data/0665a174.pdf - Use Dropout when n_hidden is large
        self.dropout_prob = 1.0                     # Dropout rate in the fully connected layer

        # CONSTANT
        self.n_features = 1                         # sprung_accel
        self.n_classes = 3                          # classifications: under, nominal, over pressure
        self.logger = logging.getLogger(__name__)   # get the logger!

        # INPUT DATA/LABELS
        self.ex_per_epoch = None
        self.training_data = None
        self.val_data = None
        self.test_data = None
        self.train_length = None

        # MODEL MEMBER VARIABLES
        self.x = None
        self.y = None
        self.cost = None
        self.accuracy = None
        self.optimizer = None
        self.trainable = None

    def preprocess_data(self):
        """Simulation data is organized by label. This method mixes and splits up the data."""
        filelists = []
        train_files = []
        val_files = []
        test_files = []
        for i in range(self.n_classes):
            modified_data_path = os.path.join(self.SIM_DATA_PATH, str(i))
            filelists.append(self._create_filename_list(modified_data_path))

            # get files for each thing
            result = self._split_datafiles(filelists[i])    # train_set, val_set, test_set
            train_files.extend(result[0])
            val_files.extend(result[1])
            test_files.extend(result[2])

        self.logger.info('Train set size: %d', len(train_files))
        self.logger.info('Validation set size: %d', len(val_files))
        self.logger.info('Test set size: %d', len(test_files))

        self.ex_per_epoch = len(train_files)
        self.training_data = self._load_data(train_files)   # features, labels
        self.val_data = self._load_data(val_files)          # features, labels
        self.test_data = self._load_data(test_files)        # features, labels
        self.train_length = self.n_epochs * self.ex_per_epoch

    def generate_batch(self, batch_idx):
        """Generate a batch and increment the sliding batch window within the data"""
        features = self.training_data[0]
        labels = self.training_data[1]

        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size

        if self.n_features > 1:
            x_batch = features[:, start_idx:end_idx]
        else:
            x_batch = features[start_idx:end_idx]

        y_batch = labels[start_idx:end_idx]

        return x_batch, y_batch

    def build_model(self):
        """ Build the CNN Model """
        with tf.name_scope("Input_Batch"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.SEQUENCE_LENGTH], name='input_data')
            self.y = tf.placeholder(tf.int64, shape=[None], name='input_labels')

            D = self.SEQUENCE_LENGTH

        """Hyperparameters"""

        initializer = tf.contrib.layers.xavier_initializer()
        self.trainable = tf.placeholder(tf.bool)          # Boolean value to guide batchnorm
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
            x_image = tf.reshape(self.x, [-1, D, 1, 1])

        self.logger.debug('Input dims: {}'.format(x_image.get_shape()))

        # CNN MODEL NOW
        """Build the graph"""
        # ewma is the decay for which we update the moving average of the
        # mean and variance in the batch-norm layers
        with tf.name_scope("Conv1"):
            w_conv1 = tf.get_variable("Conv_Layer_1",
                                      shape=[5, 1, 1, self.num_filt_1],
                                      initializer=initializer)
            b_conv1 = bias_variable([self.num_filt_1], 'bias_for_Conv_Layer_1')
            a_conv1 = conv2d(x_image, w_conv1) + b_conv1

        self.logger.debug('Conv1 output dims: %s', (a_conv1.get_shape(),))

        with tf.name_scope('Batch_norm_conv1'):
            a_conv1 = tf.contrib.layers.batch_norm(a_conv1,
                                                   is_training=self.trainable,
                                                   updates_collections=None)
            h_conv1 = tf.nn.relu(a_conv1)

        with tf.name_scope("Conv2"):
            w_conv2 = tf.get_variable("Conv_Layer_2",
                                      shape=[4, 1, self.num_filt_1, self.num_filt_2],
                                      initializer=initializer)
            b_conv2 = bias_variable([self.num_filt_2], 'bias_for_Conv_Layer_2')
            a_conv2 = conv2d(h_conv1, w_conv2) + b_conv2

        self.logger.debug('Conv2 output dims: %s', (a_conv2.get_shape(),))

        with tf.name_scope('Batch_norm_conv2'):
            a_conv2 = tf.contrib.layers.batch_norm(a_conv2,
                                                   is_training=self.trainable,
                                                   updates_collections=None)
            h_conv2 = tf.nn.relu(a_conv2)

        with tf.name_scope("Fully_Connected1"):
            w_fc1 = tf.get_variable("Fully_Connected_layer_1",
                                    shape=[D * self.num_filt_2, self.num_fc_1],
                                    initializer=initializer)
            b_fc1 = bias_variable([self.num_fc_1], 'bias_for_Fully_Connected_Layer_1')
            h_conv3_flat = tf.reshape(h_conv2, [-1, D * self.num_filt_2])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

        self.logger.debug('FCon1 output dims: %s', (h_fc1.get_shape(),))

        with tf.name_scope("Fully_Connected2"):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_prob)
            w_fc2 = tf.get_variable("w_fc2",
                                    shape=[self.num_fc_1, self.n_classes],
                                    initializer=initializer)
            b_fc2 = tf.Variable(tf.constant(0.1,
                                            shape=[self.n_classes]),
                                            name='b_fc2')
            pred = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
            tf.summary.histogram('pred', pred)

        self.logger.debug('FCon2 output dims: %s', (pred.get_shape(),))

        with tf.name_scope("Softmax"):
            # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
            # http://colah.github.io/posts/2015-09-Visual-Information/
            # https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
            # Use sparse Softmax because we have mutually exclusive classes
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.y)

            # Reduce Mean: Computes the mean of elements across dimensions of a tensor.
            # https://www.tensorflow.org/api_docs/python/tf/reduce_mean
            self.cost = tf.reduce_mean(cross_entropy, name='total')
            tf.summary.scalar('cross_entropy_loss', self.cost)

        # EVALUATE OUR MODEL
        # tf.argmax = returns index of the highest entry in a tensor along some axis.
        # So here, tf.equal is comparing predicted label to actual label, returns list of bools
        with tf.name_scope("Evaluating"):
            with tf.name_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y))
            with tf.name_scope('accuracy'):
                # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "output"))
        run_dir = os.path.abspath(os.path.join(out_dir, "trained_model_" + timestamp))
        # run_dir = os.path.abspath(os.path.join(out_dir, "trained_model"))
        checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Logging the Train Run
        self._new_run_logging(timestamp)

        """ Step 3: Train the RNN """
        with tf.Session() as sess:
            # Initialization
            bar = progressbar.ProgressBar(max_value=self.train_length)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            writer = tf.summary.FileWriter(run_dir, sess.graph)
            summary_op = tf.summary.merge_all()
            batch_idx = 0
            step = 0  # should correspond with global_step
            bar.start()
            bar.update(step)
            cost_ma = 0.0
            acc_ma = 0.0

            self.logger.info("The training shall begin.")

            # Training loop
            try:
                acc_test_before = self.evaluate_model_on_test_data(sess)

                while step <= self.train_length / self.batch_size:
                    x_batch, y_batch = self.generate_batch(batch_idx)
                    iteration = step * self.batch_size
                    if iteration % self.ex_per_epoch == 0:
                        # reset stuff for batch identifying
                        batch_idx = 0

                        # Check training performance
                        result = sess.run([self.cost, self.accuracy],
                                          feed_dict={self.x: self.training_data[0],
                                                     self.y: self.training_data[1],
                                                     self.trainable: False})
                        cost_train = result[0]
                        acc_train = result[1]

                        # Check validation performance
                        result = sess.run([self.cost, self.accuracy, summary_op],
                                          feed_dict={self.x: self.val_data[0],
                                                     self.y: self.val_data[1],
                                                     self.trainable: False})
                        cost_val = result[0]
                        acc_val = result[1]
                        summary = result[2]

                        # Check costs of training and validation models
                        if step == 0:
                            cost_ma = cost_train
                            acc_ma = acc_train
                            acc_test_before = acc_val
                        else:
                            cost_ma = 0.8 * cost_ma + 0.2 * cost_train
                            acc_ma = 0.8 * acc_ma + 0.2 * acc_train

                        # Write information to TensorBoard
                        self.logger.info('Training/Validation costs at step %d: %5.3f /  %5.3f (%5.3f)',
                                         step, cost_train, cost_val, cost_ma)
                        self.logger.info('Training/Validation accuracy at step %d: %5.3f /  %5.3f (%5.3f)',
                                         step, acc_train, acc_val, acc_ma)
                        saver.save(sess, checkpoint_prefix, global_step=step)
                        writer.flush()  # makes sure Python writes the summaries to the logfile
                        writer.add_summary(summary, step)

                    # Training step.
                    sess.run(self.optimizer, feed_dict={self.x: x_batch,
                                                        self.y: y_batch,
                                                        self.trainable: False})
                    # Update progress bar and iterate step
                    bar.update(iteration)
                    step += 1
            except KeyboardInterrupt:
                self.logger.info('Keyboard Interrupt? Gracefully quitting')
            finally:
                step -= 1
                bar.finish()
                acc_test_after = self.evaluate_model_on_test_data(sess)
                self.logger.info("The training is done.")
                self.logger.info('Test accuracy before training: %.3f', acc_test_before)
                self.logger.info('Test accuracy after training: %.3f.', acc_test_after)
                writer.close()

    def evaluate_model_on_test_data(self, sess):
        """Evaluate the model on the test data

        Returns:
            TYPE: Description
        """
        acc = sess.run(self.accuracy, feed_dict={self.x: self.test_data[0],
                                                 self.y: self.test_data[1],
                                                 self.trainable: False})

        return acc

    @staticmethod
    def reset_model():
        """Reset the model to prepare for next run."""
        tf.reset_default_graph()

    ''' Helper Functions '''
    @staticmethod
    def _create_filename_list(data_dir):
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
    def _split_datafiles(data, val_size=0.2, test_size=0.2):
        """Spit all the data we have into training, validating, and test sets.

        By default, 50/25/25 split
        Credit: https://www.slideshare.net/TaegyunJeon1/electricity-price-forecasting-with-recurrent-neural-networks

        Args:
            data (list): list of filenames
            val_size (float, optional): Percentage of data to be used for validation set
            test_size (float, optional): Percentage to data set to be used for test set

        Returns:
            train_set (list): list of training example filenames
            val_set (list): list of validation example filenames
            test_set (list): list of test example filenames
        """
        val_length = int(len(data) * val_size)
        test_length = int(len(data) * test_size)

        np.random.shuffle(data)

        val_set = data[:val_length]
        test_set = data[val_length:val_length + test_length]
        train_set = data[val_length + test_length:]

        return train_set, val_set, test_set

    def _load_data(self, filenames):
        features, labels = [], []
        for example_file in filenames:
            example_data = np.loadtxt(example_file, delimiter=',')
            if self.n_features > 1:
                ex_label = example_data[0, 0]
                ex_feature = example_data[:, 1:]
            else:
                ex_label = example_data[0]
                ex_feature = example_data[1:]

            features.append(ex_feature)
            labels.append(ex_label)

        features = np.vstack(features)

        return features, labels

    def _new_run_logging(self, timestamp):
        self.logger.info('*** NEW RUN ***')
        self.logger.info('filename: %s', "trained_model_" + timestamp)
        self.logger.info('n_epochs: %d', self.n_epochs)
        self.logger.info('batch_size: %d', self.batch_size)
        self.logger.info('exp_decay_enabled: %r', self.exp_decay_enabled)
        if self.exp_decay_enabled:
            self.logger.info('  exp_lr_starter_val: %.3f', self.exp_lr_starter_val)
            self.logger.info('  exp_lr_decay_steps: %.3f', self.exp_lr_decay_steps)
            self.logger.info('  exp_lr_decay_rate: %.3f', self.exp_lr_decay_rate)
        else:
            self.logger.info('static_lr_val: %.3f', self.static_lr_val)
        self.logger.info('Dropout Prob: %.2f', self.dropout_prob)


def main():
    """Sup Main."""
    A = CNNModel()
    A.preprocess_data()
    A.build_model()
    A.train_model()
    A.reset_model()


if __name__ == '__main__':
    # create logger with 'spam_application'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOGFILE_NAME)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(DEFAULT_FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    main()
