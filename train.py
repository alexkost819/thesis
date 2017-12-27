"""Created on 24 June 2017.
@author: Alex Kost
@description: mastermind train script for model

Attributes:
    DEFAULT_FORMAT (str): Logging format
    LOGFILE_NAME (str): Logging file name
    OUTPUT_DIR (str): TensorBoard output directory
"""

# Basic Python
import logging
import os
from time import strftime
from math import ceil

# Extended Python
import progressbar
import tensorflow as tf

# Alex Python
from data_processor import DataProcessor
from main_rnn import LSTMModel    # RNN MODEL
from main_cnn import CNNModel     # CNN MODEL

# Progressbar config
progressbar.streams.wrap_stderr()

# Logging constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'main.log'
OUTPUT_DIR = 'output'


class TrainModel(DataProcessor):
    """
    TrainModel is a class that builds and trains a provided model.

    Attributes:
        batch_size (int): number of examples in a single batch
        dropout_rate (float): dropout rate; 0.1 == 10% of input units drop out
        learning_rate (float): learning rate, used for optimizing
        logger (logger object): logging object to write to stream/file
        model (TensorFlow model object): Model to train and evaluate
        n_checks (int): number of times to check performance while training
        n_epochs (int): number of times we go through all data
        summary_op (TensorFlow operation): summary operation of all tf.summary objects
    """

    def __init__(self, model, n_epochs=20, batch_size=32, learning_rate=.00005, dropout_rate=0.5):
        """Constructor.

        Args:
            model (TensorFlow model object): Model to train and evaluate
            n_epochs (int): number of times we go through all data
            batch_size (int): number of examples in a single batch
            learning_rate (float): learning rate, used for optimizing
            dropout_rate (float): dropout rate; 0.1 == 10% of input units drop out
        """
        # TRAINING PARAMETERS
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # CONSTANT
        self.model = model(self.learning_rate, self.dropout_rate)
        self.model.build_model()
        self.summary_op = None
        self.logger = logging.getLogger(__name__)
        self.n_checks = 10

        # INPUT DATA/LABELS
        super(TrainModel, self).__init__(self.model.n_classes, self.model.n_features)
        self.preprocess_data_by_label()

        # HELPER VARIABLES
        self._ex_per_epoch = len(self.train_files)
        self._steps_per_epoch = int(ceil(self._ex_per_epoch / float(self.batch_size)))
        self._train_length_ex = self._ex_per_epoch * self.n_epochs
        self._train_length_steps = self._steps_per_epoch * self.n_epochs

    def train_model(self):
        """Train the model."""

        # SETUP TENSORBOARD FOR NEW RUN
        checkpoint_prefix, run_dir = self._setup_tensorboard_and_log_new_run()
        saver = tf.train.Saver(tf.global_variables())
        self.summary_op = tf.summary.merge_all()

        # TRAIN
        with tf.Session() as sess:
            # Initialization
            progress_bar = progressbar.ProgressBar(max_value=self._train_length_ex)
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(run_dir + '/train', sess.graph)
            val_writer = tf.summary.FileWriter(run_dir + '/val')
            batch_idx = 0
            epoch_count = 0
            progress_bar.start()
            progress_bar.update(0)

            self.logger.info("The training shall begin.")
            try:
                _, acc_test_before, _ = self.evaluate_model_on_data(sess, 'test')
                for step in range(self._train_length_steps):
                    if (step % ceil(self._train_length_steps / float(self.n_checks)) == 0) or \
                       (step == self._train_length_steps - 1):
                        # Check training and validation performance
                        cost_train, acc_train, _ = self.evaluate_model_on_data(sess, 'train')
                        cost_val, acc_val, summary = self.evaluate_model_on_data(sess, 'val')

                        # Report information to user
                        self.logger.info('%d epochs elapsed.', epoch_count)
                        self.logger.info('COST:     Train: %5.3f / Val: %5.3f', cost_train, cost_val)
                        self.logger.info('ACCURACY: Train: %5.3f / Val: %5.3f', acc_train, acc_val)

                        # Save to Tensorboard
                        val_writer.add_summary(summary, step)
                        saver.save(sess, checkpoint_prefix, global_step=step)

                    # Training step
                    x_batch, y_batch = self._generate_batch(batch_idx)
                    _, summary = sess.run([self.model.optimizer, self.summary_op],
                                          feed_dict={self.model.x: x_batch,
                                                     self.model.y: y_batch,
                                                     self.model.trainable: True})

                    # Reset/incremenet batch_idx and epoch_count
                    if step % self._steps_per_epoch == 0:
                        batch_idx = 0
                        epoch_count += 1
                    else:
                        batch_idx += 1

                    # Save to Tensorboard, update progress bar
                    train_writer.add_summary(summary, step)
                    progress = step * self.batch_size if step * self.batch_size < self._train_length_ex else self._train_length_ex
                    progress_bar.update(progress)
            except KeyboardInterrupt:
                self.logger.info('Keyboard Interrupt? Gracefully quitting.')
            finally:
                progress_bar.finish()
                _, acc_test_after, _ = self.evaluate_model_on_data(sess, 'test')
                self.logger.info("The training is done.")
                self.logger.info('Test accuracy before training: %.3f.', acc_test_before)
                self.logger.info('Test accuracy after training: %.3f.', acc_test_after)
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
        """Set up TensorBoard directories and Record new run with important details

        Returns:
            checkpoint_prefix, run_dir (string, string): checkpoint prefix, output root folder
        """
        timestamp = str(strftime("%Y.%m.%d-%H.%M.%S"))
        model_type = self.model.__class__.__name__.replace('Model', '')
        model_name = timestamp + '_' + model_type
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
        self.logger.info('  learning_rate: %f', self.learning_rate)
        self.logger.info('  dropout_rate: %.2f', 0 if self.dropout_rate is None else self.dropout_rate)

        return checkpoint_prefix, run_dir

    def _generate_batch(self, batch_idx):
        """Generate a batch and increment the sliding batch window within the data"""
        features = self.train_data[0]
        labels = self.train_data[1]

        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size - 1

        # Error handling for if sliding window goes beyond data list length
        if end_idx > self._ex_per_epoch:
            end_idx = self._ex_per_epoch

        if self.n_features > 1:
            x_batch = features[:, start_idx:end_idx]
        else:
            x_batch = features[start_idx:end_idx]

        y_batch = labels[start_idx:end_idx]
        self.logger.debug('batch_idx: %d', batch_idx)
        self.logger.debug('Got training examples %d to %d', start_idx, end_idx)

        return x_batch, y_batch


def main():
    """Sup Main!"""
    CNN = TrainModel(CNNModel, n_epochs=200, batch_size=256, learning_rate=.00005, dropout_rate=0.5)
    CNN.train_model()
    CNN.reset_model()
    LSTM = TrainModel(LSTMModel, n_epochs=200, batch_size=256, learning_rate=.0005, dropout_rate=0.5)
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
