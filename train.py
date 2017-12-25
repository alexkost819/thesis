"""Created on 24 June 2017.
@author: Alex Kost
@description: mastermind train script for model

Attributes:
    DEFAULT_FORMAT (str): Logging format
    OUTPUT_DIR (str): Description
"""

# Basic Python
import logging
import os
import time
import progressbar
import tensorflow as tf
import ipdb

# Alex Python
from data_processor import DataProcessor
from main_rnn import LSTMModel    # RNN MODEL
from main_cnn import CNNModel     # CNN MODEL

# Progressbar config
progressbar.streams.wrap_stderr()

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'train.log'
OUTPUT_DIR = 'output'


class TrainModel(DataProcessor):
    """
    CNNModel is a class that builds and trains a CNN Model.
    """

    def __init__(self, model):
        """Constructor."""
        # VARYING ACROSS TESTS
        self.n_epochs = 2                           # number of times we go through all data
        self.batch_size = 32                        # number of examples in a single batch. https://arxiv.org/abs/1206.5533

        # CONSTANT
        self.model = model(learning_rate=0.00005, dropout_rate=None)
        self.model.build_model()
        self.logger = logging.getLogger(__name__)   # get the logger!
        self.n_checks = None                        # number of times to check performance during training
        if self.n_checks is None:
            self.n_checks = self.n_epochs

        # INPUT DATA/LABELS
        super(TrainModel, self).__init__(self.batch_size, self.n_epochs, self.model.n_classes, self.model.n_features)
        self.preprocess_data_by_label()

    def train_model(self):
        """Train the model."""

        # SETUP TENSORBOARD FOR NEW RUN
        checkpoint_prefix, run_dir = self._setup_tensorboard_and_log_new_run()
        saver = tf.train.Saver(tf.global_variables())
        self.summary_op = tf.summary.merge_all()

        # TRAIN
        with tf.Session() as sess:
            # Initialization
            bar = progressbar.ProgressBar(max_value=self.train_length_ex)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(run_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(run_dir + '/test')
            val_writer = tf.summary.FileWriter(run_dir + '/val')
            batch_idx = 0
            step = 0
            epoch_count = 0
            bar.start()
            bar.update(step * self.batch_size)

            self.logger.info("The training shall begin.")
            try:
                _, acc_test_before, _ = self.evaluate_model_on_data(sess, 'test')
                while step <= self.train_length_steps:
                    if step % self.steps_per_epoch == 0:
                        # reset stuff for batch identifying
                        batch_idx = 0
                        epoch_count += 1

                    if step % (self.train_length_steps / self.n_checks) == 0 and step != 0:
                        # Check training and validation performance
                        cost_train, acc_train, _ = self.evaluate_model_on_data(sess, 'train')
                        _, _, summary = self.evaluate_model_on_data(sess, 'test')
                        test_writer.add_summary(summary, step)
                        cost_val, acc_val, summary = self.evaluate_model_on_data(sess, 'val')
                        val_writer.add_summary(summary, step)

                        # Write information to TensorBoard
                        self.logger.info('%d epochs elapsed.', epoch_count)
                        self.logger.info('COST:     Train: %5.3f / Val: %5.3f', cost_train, cost_val)
                        self.logger.info('ACCURACY: Train: %5.3f / Val: %5.3f', acc_train, acc_val)

                        saver.save(sess, checkpoint_prefix, global_step=step)

                    # Training step.
                    x_batch, y_batch = self._generate_batch(batch_idx)
                    _, summary = sess.run([self.model.optimizer, self.summary_op],
                                          feed_dict={self.model.x: x_batch,
                                                     self.model.y: y_batch,
                                                     self.model.trainable: True})

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

        cost, acc, summary = sess.run([self.model.cost, self.model.accuracy, self.summary_op],
                                      feed_dict={self.model.x: dataset[0],
                                                 self.model.y: dataset[1],
                                                 self.model.trainable: False})

        return cost, acc, summary

    @staticmethod
    def reset_model():
        """Reset the model to prepare for next run."""
        tf.reset_default_graph()

    """ Helper Functions """
    def _setup_tensorboard_and_log_new_run(self):
        timestamp = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
        model_name = 'trained_' + timestamp + '_' + self.model.__class__.__name__
        out_dir = os.path.abspath(os.path.join(os.path.curdir, OUTPUT_DIR))
        run_dir = os.path.abspath(os.path.join(out_dir, model_name))
        checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Logging the Run
        self.logger.info('*** NEW RUN ***')
        self.logger.info('filename: %s', model_name)
        self.logger.info('  batch_size: %d', self.batch_size)
        self.logger.info('  n_epochs: %d', self.n_epochs)
        self.logger.info('  n_features: %d', self.n_features)

        return checkpoint_prefix, run_dir

    def _generate_batch(self, batch_idx):
        """Generate a batch and increment the sliding batch window within the data"""
        features = self.train_data[0]
        labels = self.train_data[1]

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

    @staticmethod
    def reset_model():
        """Reset the model to prepare for next run."""
        tf.reset_default_graph()


def main():
    """Sup Main."""
    CNN = TrainModel(CNNModel)
    CNN.train_model()
    CNN.reset_model()
    LSTM = TrainModel(LSTMModel)
    LSTM.train_model()
    LSTM.reset_model()


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
