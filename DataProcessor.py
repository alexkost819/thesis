"""Created on 17 December 2017.
@author: Alex Kost
@description: Main python code file for preprocessing data

Attributes:
    DEFAULT_FORMAT (str): Logging format
    SIM_DATA_PATH (str): Local simulation data output folder path
"""

import logging
import os
import numpy as np

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
SIM_DATA_PATH = 'Data/simulation_labeled'


class DataProcessor(object):
    """
    DataProcessor is a class that can process datasets in different ways
    """
    def __init__(self, batch_size, n_epochs, n_classes, n_features):

        # assign input variables
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.n_features = n_features

        # filenames
        self.train_files = None
        self.val_files = None
        self.test_files = None

        # opened data
        self.training_data = None
        self.val_data = None
        self.test_data = None

        # declare all the things
        self.ex_per_epoch = None
        self.steps_per_epoch = None
        self.train_length_ex = None
        self.train_length_steps = None

        # get the logger!
        self.logger = logging.getLogger(__name__)

    def preprocess_all_data(self):
        all_files = self._create_filename_list(SIM_DATA_PATH)
        np.random.shuffle(all_files)

        train_val_test_files = self._split_datafiles(all_files)    # train_set, val_set, test_set
        train_files = train_val_test_files[0]
        val_files = train_val_test_files[1]
        test_files = train_val_test_files[2]

        # Report sizes
        self.logger.info('Train set size: %d', len(train_files))
        self.logger.info('Validation set size: %d', len(val_files))
        self.logger.info('Test set size: %d', len(test_files))

        # Assign member variables
        self.ex_per_epoch = len(train_files)
        self.steps_per_epoch = self.ex_per_epoch / self.batch_size
        self.training_data = self._load_data(train_files)   # features, labels
        self.val_data = self._load_data(val_files)          # features, labels
        self.test_data = self._load_data(test_files)        # features, labels
        self.train_length_ex = self.n_epochs * self.ex_per_epoch
        self.train_length_steps = self.train_length_ex / self.batch_size

    def preprocess_data_by_label(self):
        """Simulation data is organized by label. This method mixes and splits up the data."""
        train_files = []
        val_files = []
        test_files = []
        for i in range(self.n_classes):
            modified_data_path = os.path.join(SIM_DATA_PATH, str(i))
            class_files = self._create_filename_list(modified_data_path)

            # get files for each thing
            result = self._split_datafiles(class_files)    # train_set, val_set, test_set
            train_files.extend(result[0])
            val_files.extend(result[1])
            test_files.extend(result[2])
            self.logger.debug('%d/%d/%d added to train/val/test set from class %d.',
                              len(result[0]), len(result[1]),
                              len(result[2]), i)

        # Shuffle data
        np.random.shuffle(train_files)
        np.random.shuffle(val_files)
        np.random.shuffle(test_files)

        # Report sizes
        self.logger.info('Train set size: %d', len(train_files))
        self.logger.info('Validation set size: %d', len(val_files))
        self.logger.info('Test set size: %d', len(test_files))

        # Assign member variables
        self.ex_per_epoch = len(train_files)
        self.steps_per_epoch = self.ex_per_epoch / self.batch_size
        self.training_data = self._load_data(train_files)   # features, labels
        self.val_data = self._load_data(val_files)          # features, labels
        self.test_data = self._load_data(test_files)        # features, labels
        self.train_length_ex = self.n_epochs * self.ex_per_epoch
        self.train_length_steps = self.train_length_ex / self.batch_size

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

    def use_all_files_for_training(self, filename_list):
        """ """
        self.ex_per_epoch = len(filename_list)
        self.train_length_ex = self.ex_per_epoch * self.n_epochs

        # TODO(akost): Finish fixing these up
        self.ex_per_epoch = len(train_files)
        self.steps_per_epoch = self.ex_per_epoch / self.batch_size
        self.training_data = self._load_data(train_files)   # features, labels
        self.val_data = self._load_data(val_files)          # features, labels
        self.test_data = self._load_data(test_files)        # features, labels
        self.train_length_ex = self.n_epochs * self.ex_per_epoch
        self.train_length_steps = self.train_length_ex / self.batch_size

    """ Helper Functions """
    def _create_filename_list(self, data_dir):
        """Identify the list of CSV files based on a given data_dir.

        Args:
            data_dir (string): local path to where the data is saved.

        Returns:
            list of strings: a list of CSV files found in the data directory
        """
        filenames = []
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".csv"):
                    rel_filepath = os.path.join(root, filename)
                    abs_filepath = os.path.abspath(rel_filepath)
                    filenames.append(abs_filepath)

        self.ex_per_epoch = len(filenames)

        return filenames

    @staticmethod
    def _split_datafiles(data, val_size=0.2, test_size=0.2):
        """Spit all the data we have into training, validating, and test sets.

        By default, 60/20/20 split
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

        val_set = data[:val_length]
        test_set = data[val_length:val_length + test_length]
        train_set = data[val_length + test_length:]

        return train_set, val_set, test_set

    def _load_data(self, filenames):
        # Get features and labels from dataset
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

        # stack features
        features = np.vstack(features)

        return features, labels
