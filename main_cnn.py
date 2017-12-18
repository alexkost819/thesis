"""Created on 24 June 2017.
@author: Alex Kost
@description: Main python code file for Applying CNN as a TPMS.

Attributes:
    DEFAULT_FORMAT (str): Logging format
    LOGFILE_NAME (str): Logging file name
    SIM_DATA_PATH (str): Local simulation data output folder path
    SIM_LENGTH_FIX (int): bias to datapoint length due to slicing ops in Matlab, datapoints
    SIM_LENGTH_SEQ (int): simulation length, datapoints
    SIM_LENGTH_TIME (float): simulation time, sec
    SIM_RESOLUTION (float): simulation resolution, sec/datapoint
"""

# Basic Python
import logging
import os
import progressbar
import time
import tensorflow as tf
import numpy as np

# Alex Python
from DataProcessor import DataProcessor

# Progressbar config
progressbar.streams.wrap_stderr()

# Logging Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'main_cnn.log'

# Simulation Constants
OUTPUT_DIR = 'output'
SIM_LENGTH_TIME = 1.5 - .45
SIM_RESOLUTION = .001
SIM_LENGTH_FIX = 2
SIM_LENGTH_SEQ = int(SIM_LENGTH_TIME / SIM_RESOLUTION) + SIM_LENGTH_FIX


class CNNModel(DataProcessor):
    """
    CNNModel is a class that builds and trains a CNN Model.
    """

    def __init__(self):
        """Constructor."""
        # VARYING ACROSS TESTS
        self.n_epochs = 100                         # number of times we go through all data
        self.batch_size = 32                        # number of examples in a single batch. https://arxiv.org/abs/1206.5533
        self.num_filt_1 = 16                        # number of filters in first conv layer
        self.num_filt_2 = 14                        # number of filters in second conv layer
        self.num_fc_1 = 40                          # number of neurons in fully connected layer
        #self.dropout_rate = 0.99                   # dropout rate, between 0 & 1. Ex) 0.1 == 10% of input units drop out

        # LEARNING RATE
        # Exponential Decay Parameters
        # https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
        self.static_lr_val = 0.00005
        self.exp_decay_enabled = False             # enable/disable exponentially decaying learning rate
        self.exp_lr_starter_val = .00005
        self.exp_lr_decay_steps = 633 * 10
        self.exp_lr_decay_rate = 0.96

        # CONSTANT
        self.n_features = 1                         # sprung_accel
        self.n_classes = 3                          # classifications: under, nominal, over pressure
        self.logger = logging.getLogger(__name__)   # get the logger!
        self.n_checks = 20                          # number of times to check performance during training

        # INPUT DATA/LABELS
        super(CNNModel, self).__init__(self.batch_size, self.n_epochs, self.n_classes, self.n_features)
        self.preprocess_data_by_label()

        # MODEL MEMBER VARIABLES
        self.x = None
        self.y = None
        self.cost = None
        self.accuracy = None
        self.optimizer = None
        self.trainable = tf.placeholder(tf.bool, name='trainable')     # Boolean value to guide batchnorm
                                                    # Set false when evaluating, set true when training
        self.summary_op = None                      # summary operation to write data

    def generate_batch(self, batch_idx):
        """Generate a batch and increment the sliding batch window within the data"""
        features = self.training_data[0]
        labels = self.training_data[1]

        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size - 1

        # Error handling for if sliding window goes beyond data list length
        if end_idx > self.ex_per_epoch:
            end_idx -= (end_idx % self.ex_per_epoch)

        if self.n_features > 1:
            x_batch = features[:, start_idx:end_idx]
        else:
            x_batch = features[start_idx:end_idx]

        y_batch = labels[start_idx:end_idx]
        self.logger.debug('batch_idx: %d', batch_idx)
        self.logger.debug('Got training examples %d to %d', start_idx, end_idx)

        return x_batch, y_batch

    def build_model(self):
        """ Build the CNN Model """
        self.x = tf.placeholder(tf.float32, shape=[None, SIM_LENGTH_SEQ], name='input_data')
        self.y = tf.placeholder(tf.int64, shape=[None], name='input_labels')

        with tf.name_scope("Reshape_Data"):
            # tf.nn.conv2d requires inputs to be shaped as follows:
            # [batch, in_height, in_width, in_channels]
            # so -1 = batch size, should adapt accordingly
            # in_height = "height" of the image (so one dimension)
            # in_width = "width" of image
            x_image = tf.reshape(self.x, [-1, SIM_LENGTH_SEQ, 1, self.n_features])

        self.logger.debug('Input dims: {}'.format(x_image.get_shape()))

        with tf.variable_scope("ConvBatch1"):
            conv1 = tf.contrib.layers.conv2d(inputs=x_image,
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
            #fc1_dropout = tf.layers.dropout(fc1, self.dropout_rate, name='Dropout')
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
        # tf.reduce_mean = reduces tensor to mean scalar value of tensor
        with tf.name_scope("Softmax"):
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
        with tf.name_scope("Evaluating"):
                correct_pred = tf.equal(tf.argmax(pred, 1), self.y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

        # OPTIMIZE OUR MODEL
        # either use exponentially decaying learning rate or static learning rate with
        #     AdamOptimizer to minimize the cost
        with tf.name_scope("Optimizing"):
            if self.exp_decay_enabled:
                global_step = tf.get_variable('global_step',
                                              shape=(),
                                              initializer=tf.zeros_initializer(),
                                              trainable=False)
                learning_rate = tf.train.exponential_decay(learning_rate=self.exp_lr_starter_val,
                                                           global_step=global_step,
                                                           decay_steps=self.exp_lr_decay_steps,
                                                           decay_rate=self.exp_lr_decay_rate,
                                                           staircase=True,
                                                           name='learning_rate')

                # Passing global_step to minimize() will increment it at each step.
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                        name='train').minimize(self.cost,
                                                                               global_step=global_step)

            else:
                learning_rate = tf.constant(self.static_lr_val)
                self.optimizer = tf.train.AdamOptimizer(learning_rate, name='train').minimize(self.cost)

            tf.summary.scalar('learning_rate', learning_rate)

    def train_model(self):
        """Train the model."""

        """ Step 9: Set up Tensorboard """
        timestamp = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, OUTPUT_DIR))
        run_dir = os.path.abspath(os.path.join(out_dir, "trained_model_" + timestamp))
        # run_dir = os.path.abspath(os.path.join(out_dir, "trained_model"))
        checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        self.summary_op = tf.summary.merge_all()

        # Logging the Train Run
        self._new_run_logging(timestamp)

        """ Step 1: Train the RNN """
        with tf.Session() as sess:
            # Initialization
            bar = progressbar.ProgressBar(max_value=self.train_length_ex)
            sess.run(tf.global_variables_initializer())
#            sess.run(tf.local_variables_initializer())
            train_writer = tf.summary.FileWriter(run_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(run_dir + '/test')
            val_writer = tf.summary.FileWriter(run_dir + '/val')
            batch_idx = 0
            step = 0  # should correspond with global_step
            bar.start()
            bar.update(step * self.batch_size)

            self.logger.info("The training shall begin.")
            try:
                _, acc_test_before, _ = self.evaluate_model_on_data(sess, 'test')
                while step <= self.train_length_steps:
                    if step % self.steps_per_epoch == 0:
                        # reset stuff for batch identifying
                        batch_idx = 0

                    if step % (self.train_length_steps / self. n_checks) == 0 and step != 0:
                        # Check training and validation performance
                        cost_train, acc_train, _ = self.evaluate_model_on_data(sess, 'train')
                        _, _, summary = self.evaluate_model_on_data(sess, 'test')
                        test_writer.add_summary(summary, step)
                        cost_val, acc_val, summary = self.evaluate_model_on_data(sess, 'val')
                        val_writer.add_summary(summary, step)

                        # Write information to TensorBoard
                        self.logger.info('%d epochs elapsed.', step * self.batch_size / self.ex_per_epoch)
                        self.logger.info('COST:     Train: %5.3f / Val: %5.3f', cost_train, cost_val)
                        self.logger.info('ACCURACY: Train: %5.3f / Val: %5.3f', acc_train, acc_val)

                        saver.save(sess, checkpoint_prefix, global_step=step)

                    # Training step.
                    x_batch, y_batch = self.generate_batch(batch_idx)
                    _, summary = sess.run([self.optimizer, self.summary_op],
                                          feed_dict={self.x: x_batch,
                                                     self.y: y_batch,
                                                     self.trainable: True})

                    # Update progress bar and iterate step/batch_idx
                    train_writer.add_summary(summary, step)
                    bar.update(step * self.batch_size)
                    batch_idx += 1
                    step += 1
            except KeyboardInterrupt:
                self.logger.info('Keyboard Interrupt? Gracefully quitting.')
            finally:
                step -= 1
                bar.finish()
                _, acc_test_after, _ = self.evaluate_model_on_data(sess, 'test')
                self.logger.info("The training is done.")
                self.logger.info('Test accuracy before training: %.3f.', acc_test_before)
                self.logger.info('Test accuracy after training: %.3f.', acc_test_after)
                test_writer.close()
                train_writer.close()
                val_writer.close()

    def evaluate_model_on_data(self, sess, dataset_label):
        """Evaluate the model on the entire training data.

        Args:
            sess (tf.Session object): active session object
            dataset_label (string): dataset label

        Returns:
            float, float: the cost and accuracy of the model based on the dataset.
        """
        try:
            dataset_dict = {'test': self.test_data,
                            'train': self.test_data,
                            'val': self.val_data}
            dataset = dataset_dict[dataset_label]
        except KeyError:
            raise '"dataset" arg must be in dataset dict: {}'.format(dataset_dict.keys())

        cost, acc, summary = sess.run([self.cost, self.accuracy, self.summary_op],
                                      feed_dict={self.x: dataset[0],
                                                 self.y: dataset[1],
                                                 self.trainable: False})

        return cost, acc, summary

    @staticmethod
    def reset_model():
        """Reset the model to prepare for next run."""
        tf.reset_default_graph()

    def _new_run_logging(self, timestamp):
        self.logger.info('*** NEW RUN ***')
        self.logger.info('filename: %s', "trained_model_" + timestamp)
        self.logger.info('n_epochs: %d', self.n_epochs)
        self.logger.info('batch_size: %d', self.batch_size)
        self.logger.info('exp_decay_enabled: %r', self.exp_decay_enabled)
        if self.exp_decay_enabled:
            self.logger.info('  exp_lr_starter_val: %f', self.exp_lr_starter_val)
            self.logger.info('  exp_lr_decay_steps: %d', self.exp_lr_decay_steps)
            self.logger.info('  exp_lr_decay_rate: %.2f', self.exp_lr_decay_rate)
        else:
            self.logger.info('static_lr_val: %f', self.static_lr_val)
        #self.logger.info('Dropout Rate: %.2f', self.dropout_rate)


def main():
    """Sup Main."""
    A = CNNModel()
    A.build_model()
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
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(DEFAULT_FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    main()
