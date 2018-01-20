"""Created on 24 June 2017.
@author: Alex Kost
@description: Training class for CNN and RNN models

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
from rnn_model import RNNModel     # RNN MODEL
from cnn_model import CNNModel     # CNN MODEL

# Progressbar config
progressbar.streams.wrap_stderr()

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'train.log'
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
    def __init__(self, model, n_epochs=20, batch_size=32):
        """Constructor.

        Args:
            model (TensorFlow model object): Model to train and evaluate
            n_epochs (int, optional): number of times we go through all data
            batch_size (int, optional): number of examples in a single batch
        """
        # TRAINING PARAMETERS
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # CONSTANT
        self.model = model
        self.summary_op = None
        self.logger = logging.getLogger(__name__)
        self.n_checks = 5

        # INPUT DATA/LABELS
        super(TrainModel, self).__init__(self.model.n_classes, self.model.n_features)
        self.preprocess_data_by_label()

        # HELPER VARIABLES
        self._ex_per_epoch = None
        self._steps_per_epoch = None
        self._train_length_ex = None
        self._train_length_steps = None
        self.calculate_helpers()

    def calculate_helpers(self):
        """Calculate helper variables for training length."""
        self._ex_per_epoch = len(self.train_files)
        self._steps_per_epoch = int(ceil(self._ex_per_epoch / float(self.batch_size)))
        self._train_length_ex = self._ex_per_epoch * self.n_epochs
        self._train_length_steps = self._steps_per_epoch * self.n_epochs

        self.logger.debug('self._ex_per_epoch: %d', self._ex_per_epoch)
        self.logger.debug('self._steps_per_epoch: %d', self._steps_per_epoch)
        self.logger.debug('self._train_length_ex: %d', self._train_length_ex)
        self.logger.debug('self._train_length_steps: %d', self._train_length_steps)

    def train_model(self, use_tensorboard=True):
        """Train the model.

        Args:
            use_tensorboard (bool, optional): Description

        Returns:
            TYPE: Description
        """

        # SETUP TENSORBOARD FOR NEW RUN
        if use_tensorboard:
            checkpoint_prefix, run_dir = self._setup_tensorboard_directories()
            saver = tf.train.Saver(tf.global_variables())
        else:
            self.logger.info('*** NEW RUN ***')
        self._log_training_and_model_params()
        self.summary_op = tf.summary.merge_all()

        # TRAIN
        with tf.Session() as sess:
            # Initialization
            progress_bar = progressbar.ProgressBar(max_value=self._train_length_steps)
            sess.run(tf.global_variables_initializer())
            if use_tensorboard:
                train_writer = tf.summary.FileWriter(run_dir + '/train', sess.graph)
                val_writer = tf.summary.FileWriter(run_dir + '/val')
            batch_idx = 0
            progress_bar.start()
            progress_bar.update(0)

            self.logger.info("The training shall begin.")
            try:
                _, acc_test_before, _ = self.evaluate_model_on_data(sess, 'test')
                for step in range(self._train_length_steps):
                    # Reset/increment batch_idx
                    if step % self._steps_per_epoch == 0:
                        batch_idx = 0
                    else:
                        batch_idx += 1

                    if use_tensorboard:
                        do_full_eval = step % ceil(self._train_length_steps / float(self.n_checks)) == 0
                        do_full_eval = do_full_eval or (step == self._train_length_steps - 1)
                        if do_full_eval:
                            # Check training and validation performance
                            cost_train, acc_train, _ = self.evaluate_model_on_data(sess, 'train')
                            cost_val, acc_val, summary = self.evaluate_model_on_data(sess, 'val')

                            # Report information to user
                            self.logger.info('%d epochs elapsed.', step / self._steps_per_epoch)
                            self.logger.info('COST:     Train: %5.3f / Val: %5.3f', cost_train, cost_val)
                            self.logger.info('ACCURACY: Train: %5.3f / Val: %5.3f', acc_train, acc_val)

                            # Save to Tensorboard
                            val_writer.add_summary(summary, step)
                            saver.save(sess, checkpoint_prefix, global_step=step)

                            # # If model is not learning immediately, break out of training
                            # if acc_val == acc_test_before and step > 100:
                            #     self.logger.info('Stuck on value: %d', acc_val)
                            #     break

                    # Training step
                    x_batch, y_batch = self._generate_batch(batch_idx)
                    _, summary = sess.run([self.model.optimizer, self.summary_op],
                                          feed_dict={self.model.x: x_batch,
                                                     self.model.y: y_batch,
                                                     self.model.trainable: True})

                    # Save to Tensorboard, update progress bar
                    if use_tensorboard:
                        train_writer.add_summary(summary, step)
                    progress_bar.update(step)
            except KeyboardInterrupt:
                self.logger.info('Keyboard Interrupt? Gracefully quitting.')
            finally:
                progress_bar.finish()
                _, acc_test_after, _ = self.evaluate_model_on_data(sess, 'test')
                self.logger.info("The training is done.")
                self.logger.info('Test accuracy before training: %.3f.', acc_test_before)
                self.logger.info('Test accuracy after training: %.3f.', acc_test_after)
                if use_tensorboard:
                    train_writer.close()
                    val_writer.close()

        return acc_test_after

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
    def _setup_tensorboard_directories(self):
        """Set up TensorBoard directories.

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

        return checkpoint_prefix, run_dir

    def _log_training_and_model_params(self):
        """Record new run details."""
        model_type = self.model.__class__.__name__

        self.logger.info('  *** TRAINING ***')
        self.logger.info('    n_epochs: %d', self.n_epochs)
        self.logger.info('    batch_size: %d', self.batch_size)
        self.logger.info('  *** MODEL ***')
        if 'CNN' in model_type:
            self.logger.info('    num_filt_1: %d', self.model.num_filt_1)
            self.logger.info('    kernel_size: %d', self.model.kernel_size)
            self.logger.info('    num_fc_1: %d', self.model.num_fc_1)
        elif 'RNN' in model_type:
            self.logger.info('    n_hidden: %d', self.model.n_hidden)
            self.logger.info('    num_fc_1: %d', self.model.num_fc_1)
            self.logger.info('    n_layers: %d', self.model.n_layers)

        self.logger.info('    dropout_rate: %f', self.model.dropout_rate)
        self.logger.info('    learning_rate: %f', self.model.learning_rate)
        self.logger.info('    beta1: %f', self.model.beta1)
        self.logger.info('    beta2: %f', self.model.beta2)
        self.logger.info('    epsilon: %f', self.model.epsilon)

    def _generate_batch(self, batch_idx):
        """Generate a batch and increment the sliding batch window within the data."""
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
    models = [CNNModel(), RNNModel()]
    for model in models:
        model.build_model()
        train = TrainModel(model, n_epochs=200, batch_size=128)
        train.train_model()
        train.reset_model()


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
