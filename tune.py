"""Created on 6 Jan 2017.
@author: Alex Kost
@description: mastermind tuning script for model

Attributes:
    DEFAULT_FORMAT (str): Logging format
    LOGFILE_NAME (str): Logging file name
    OUTPUT_DIR (str): TensorBoard output directory
"""

# Basic Python
import logging

# Extended Python
from sigopt import Connection

# Alex Python
from train import TrainModel
from rnn_model import RNNModel     # RNN MODEL
from cnn_model import CNNModel     # CNN MODEL

# Constants
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
LOGFILE_NAME = 'tune.log'
#EXPERIMENT_ID = 34189           # CNNModel Accuracy v1
#EXPERIMENT_ID = 34205           # CNNModel Accuracy v2
#EXPERIMENT_ID = 34424           # CNNModel Accuracy v3

EXPERIMENT_ID = 34631            # RNNModel Accuracy v1


class SigOptTune(object):
    def __init__(self):
        """Constructor."""
        self.logger = logging.getLogger(__name__)   # get the logger!

        self.conn = Connection(client_token="XWCROUDALHMNJFABTLYVXBUHISZQKKACUGULCENHPSZNQPSD")
        self.conn.set_api_url("https://api.sigopt.com")
        self.experiment = None
        self.suggestion = None

        self.model = None
        self.acc = None

    def create_cnn_experiment(self):
        """Create experiment. Modify as needed."""
        self.experiment = self.conn.experiments().create(
            name="CNNModel Accuracy v3",
            parameters=[dict(name="learning_rate",
                             bounds=dict(min=0.00001, max=0.1),
                             type="double"),
                        dict(name="dropout_rate",
                             bounds=dict(min=0.2, max=0.9),
                             type="double"),
                        dict(name="beta1",
                             bounds=dict(min=0.0001, max=0.999),
                             type="double"),
                        dict(name="beta2",
                             bounds=dict(min=0.0001, max=0.999),
                             type="double"),
                        dict(name="epsilon",
                             bounds=dict(min=1e-8, max=1.0),
                             type="double"),
                        dict(name="num_filt_1",
                             bounds=dict(min=1, max=40),
                             type="int"),
                        dict(name="kernel_size",
                             bounds=dict(min=1, max=10),
                             type="int"),
                        dict(name="num_fc_1",
                             bounds=dict(min=1, max=40),
                             type="int")
                        ])

        self.logger.info('Experiment created! ID %d.', self.experiment.id)

    def create_rnn_experiment(self):
        """Create experiment. Modify as needed."""
        self.experiment = self.conn.experiments().create(
            name="RNNModel Accuracy v1",
            parameters=[dict(name="learning_rate",
                             bounds=dict(min=0.00001, max=0.1),
                             type="double"),
                        dict(name="dropout_rate",
                             bounds=dict(min=0.2, max=0.9),
                             type="double"),
                        dict(name="beta1",
                             bounds=dict(min=0.0001, max=0.999),
                             type="double"),
                        dict(name="beta2",
                             bounds=dict(min=0.0001, max=0.999),
                             type="double"),
                        dict(name="epsilon",
                             bounds=dict(min=1e-8, max=1.0),
                             type="double"),
                        dict(name="n_hidden",
                             bounds=dict(min=1, max=40),
                             type="int"),
                        dict(name="num_fc_1",
                             bounds=dict(min=1, max=40),
                             type="int"),
                        dict(name="n_layers",
                             bounds=dict(min=1, max=10),
                             type="int")
                        ])

        self.logger.info('Experiment created! ID %d.', self.experiment.id)

    def get_suggestions(self):
        """Create suggestions for next iteration."""
        try:
            self.suggestion = self.conn.experiments(EXPERIMENT_ID).suggestions().create()
            logger.info('Created new suggestions.')
        except:
            self.conn.experiments(EXPERIMENT_ID).suggestions().delete(state="open")
            self.suggestion = self.conn.experiments(EXPERIMENT_ID).suggestions().create()
            logger.info('Deleted old and created new suggestions.')

    def update_parameters(self):
        """Update model parameters with suggestions."""
        #model_type = self.model.__class__.__name__.replace('Model', '')

        params = self.suggestion.assignments
        # if model_type == 'CNN':
        #     self.model.num_filt_1 = int(params['num_filt_1'])
        #     self.model.kernel_size = int(params['kernel_size'])
        #     self.model.num_fc_1 = int(params['num_fc_1'])
        # elif model_type == 'RNN':
        #     self.model.n_hidden = int(params['n_hidden'])
        #     self.model.num_fc_1 = int(params['num_fc_1'])
        #     self.model.n_layers = int(params['n_layers'])

        #self.model.dropout_rate = params['dropout_rate']
        self.model.learning_rate = params['learning_rate']
        self.model.beta1 = params['beta1']
        self.model.beta2 = params['beta2']
        self.model.epsilon = params['epsilon']

    def report_observation(self):
        """Report observation to SigOpt."""
        self.conn.experiments(EXPERIMENT_ID).observations().create(
                suggestion=self.suggestion.id,
                value=float(self.acc),
                value_stddev=0.05)

    def optimization_loop(self, model):
        """Optimize the parameters based on suggestions."""
        for i in range(100):
            self.logger.info('Optimization Loop Count: %d', i)

            # assign suggestions to parameters and hyperparameters
            self.get_suggestions()

            # update model class
            self.model = model()
            self.update_parameters()
            self.model.build_model()

            # update training class
            train = TrainModel(self.model, n_epochs=200, batch_size=128)

            # run the training stuff
            self.acc = train.train_model()
            train.reset_model()

            # report to SigOpt
            self.report_observation()


class GridSearchTune(object):
    def __init__(self):
        """Constructor."""
        self.logger = logging.getLogger(__name__)   # get the logger!

    def tune_cnn_with_gridsearch():
        """Grid search to identify best hyperparameters for CNN model."""
        cnn_model_values = []
        n_epoch_list = [100, 200, 300, 400, 500]                                # 5
        batch_size_list = [16, 32, 64, 128, 256]                                # 5
        learning_rate_list = [.0001, .0005, .00001, .00005]                     # 4
        dropout_rate_list = [0.2, 0.5, 0.7]                                     # 3

        try:
            for n_epoch in n_epoch_list:
                for batch_size in batch_size_list:
                    for learning_rate in learning_rate_list:
                        for dropout_rate in dropout_rate_list:
                            for num_filt_1 in [8, 16, 32]:              # CNN ONLY  # 3
                                for num_filt_2 in [10, 20, 30, 40]:     # CNN ONLY  # 4
                                    for num_fc_1 in [10, 20, 30, 40]:   # CNN ONLY  # 4
                                        CNN = TrainModel(CNNModel, n_epoch, batch_size, learning_rate, dropout_rate)
                                        CNN.model.num_filt_1 = num_filt_1
                                        CNN.model.num_filt_2 = num_filt_2
                                        CNN.model.num_fc_1 = num_fc_1
                                        CNN.model.build_model()
                                        CNN.calculate_helpers()
                                        acc = CNN.train_model()
                                        CNN.reset_model()

                                        results = [acc, n_epoch, batch_size, learning_rate, dropout_rate, num_filt_1, num_filt_2, num_fc_1]
                                        cnn_model_values.append(results)
        except:
            pass
        finally:
            best_cnn_run = max(cnn_model_values, key=lambda x: x[0])
            logger.info('Best CNN run: {}'.format(best_cnn_run))
            logger.info('All CNN runs: {}'.format(cnn_model_values))

    def tune_rnn_with_gridsearch():
        """Grid search to identify best hyperparameters for RNN."""
        rnn_model_values = []
        n_epoch_list = [200, 400, 600, 800, 1000]                               # 5
        batch_size_list = [16, 32, 64, 128, 256]                                # 5
        learning_rate_list = [.001, .005, .0001, .0005]                         # 4
        dropout_rate_list = [0.2, 0.5, 0.7]                                     # 3

        for n_epoch in n_epoch_list:
            for batch_size in batch_size_list:
                for learning_rate in learning_rate_list:
                    for dropout_rate in dropout_rate_list:
                        for n_hidden in [8, 16, 32]:                # RNN ONLY
                            for num_fc_1 in [10, 20, 30, 40]:       # RNN ONLY
                                for n_layers in [1, 2, 3]:          # RNN ONLY
                                    RNN = TrainModel(RNNModel, n_epoch, batch_size, learning_rate, dropout_rate)
                                    RNN.model.n_hidden = n_hidden
                                    RNN.model.num_fc_1 = num_fc_1
                                    RNN.model.n_layers = n_layers

                                    RNN.model.build_model()
                                    RNN.calculate_helpers()
                                    acc = RNN.train_model()
                                    RNN.reset_model()

                                    rnn_model_values.append([acc, n_epoch, batch_size, learning_rate, dropout_rate, n_hidden, num_fc_1, n_layers])

            best_rnn_run = max(rnn_model_values, key=lambda x: x[0])
            logger.info('Best RNN run: {}'.format(best_rnn_run))
            logger.info('All RNN runs: {}'.format(rnn_model_values))


def main():
    """Sup Main!"""
    tune = SigOptTune()
    #tune.create_cnn_experiment()
    #tune.optimization_loop(CNNModel)
    #tune.create_rnn_experiment()
    tune.optimization_loop(RNNModel)

if __name__ == '__main__':
    # create logger with 'spam_application'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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
