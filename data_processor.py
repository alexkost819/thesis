"""Created on 17 December 2017.
@author: Alex Kost
@description: Main python code file for preprocessing data

Attributes:
    SIM_DATA_PATH (str): Local simulation data output folder path
    SIM_LENGTH_FIX (int): bias to datapoint length due to slicing ops in Matlab, datapoints
    SIM_LENGTH_SEQ (int): simulation length, datapoints
    SIM_LENGTH_TIME (float): simulation time, sec
    SIM_RESOLUTION (float): simulation resolution, sec/datapoint
"""

# Basic Python
import logging
import os

# Extended Python
import numpy as np

# Simulation Constants
SIM_LENGTH_TIME = 1.5 - .45
SIM_RESOLUTION = .001
SIM_LENGTH_FIX = 2
SIM_LENGTH_SEQ = int(SIM_LENGTH_TIME / SIM_RESOLUTION) + SIM_LENGTH_FIX
SIM_DATA_PATH = 'Data/simulation_labeled'


class DataProcessor(object):
    """
    DataProcessor is a class that processes datasets.
    """
    def __init__(self, n_classes, n_features):
        """Constructor

        Args:
            n_classes (int): label classifications
            n_features (int): features per example
        """
        # assign input variables
        self.n_classes = n_classes
        self.n_features = n_features

        # FILENAME LISTS
        self.train_files = []
        self.val_files = []
        self.test_files = []

        # LOADED DATA
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.logger = logging.getLogger(__name__)   # get the logger!

    def preprocess_all_data(self):
        """Shuffle all data and then preprocess the files."""
        all_files = self._create_filename_list(SIM_DATA_PATH)
        np.random.shuffle(all_files)

        train_val_test_files = self._split_datafiles(all_files)    # train_set, val_set, test_set
        self.train_files = train_val_test_files[0]
        self.val_files = train_val_test_files[1]
        self.test_files = train_val_test_files[2]

        # Report sizes and load all datasets
        self.logger.info('Train set size: %d', len(self.train_files))
        self.logger.info('Validation set size: %d', len(self.val_files))
        self.logger.info('Test set size: %d', len(self.test_files))
        self._load_all_datasets()

    def preprocess_data_by_label(self):
        """Simulation data is organized by label. This method mixes and splits up the data."""
        for i in range(self.n_classes):
            modified_data_path = os.path.join(SIM_DATA_PATH, str(i))
            class_files = self._create_filename_list(modified_data_path)

            # get files for each thing
            result = self._split_datafiles(class_files)    # train_set, val_set, test_set
            self.train_files.extend(result[0])
            self.val_files.extend(result[1])
            self.test_files.extend(result[2])
            self.logger.debug('%d/%d/%d added to train/val/test set from class %d.',
                              len(result[0]), len(result[1]),
                              len(result[2]), i)

        # Shuffle data
        np.random.shuffle(self.train_files)
        np.random.shuffle(self.val_files)
        np.random.shuffle(self.test_files)

        # Report sizes and load all datasets
        self.logger.info('Train set size: %d', len(self.train_files))
        self.logger.info('Validation set size: %d', len(self.val_files))
        self.logger.info('Test set size: %d', len(self.test_files))
        self._load_all_datasets()

    """ Helper Functions """
    def _load_all_datasets(self):
        """Assign class member variables after processing filenames."""
        self.train_data = self._load_data(self.train_files)   # features, labels
        self.val_data = self._load_data(self.val_files)          # features, labels
        self.test_data = self._load_data(self.test_files)        # features, labels

    def _create_filename_list(self, data_dir):
        """Identify the list of CSV files based on a given data_dir.

        Args:
            data_dir (string): local path to where the data is saved.

        Returns:
            filenames (list of strings): a list of CSV files found in the data directory
        """
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
        """Load data from the filenames

        Args:
            filenames (list of strings): filenames

        Returns:
            features, labels (np.array, np.array): loaded features and labels
        """
        # Get features and labels from dataset
        features, labels = [], []
        for example_file in filenames:
            example_data = np.loadtxt(example_file, delimiter=',')

            ex_label = example_data[0, 0] if self.n_features > 1 else example_data[0]
            ex_feature = example_data[:, 1:] if self.n_features > 1 else example_data[1:]

            features.append(ex_feature)
            labels.append(ex_label)

        # stack features
        features = np.vstack(features)

        return features, labels
