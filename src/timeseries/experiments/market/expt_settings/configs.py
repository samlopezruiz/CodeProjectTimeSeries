import os
from timeseries.experiments.market.data_formatter.snp import SnPFormatter


class ExperimentConfig(object):
    """Defines experiment configs and paths to outputs.

  Attributes:
    root_folder: Root folder to contain all experimental outputs.
    experiment: Name of experiment to run.
    data_folder: Folder to store data for experiment.
    model_folder: Folder to store serialised models.
    results_folder: Folder to store results.
    data_csv_path: Path to primary data csv file used in experiment.
    hyperparam_iterations: Default number of random search iterations for
      experiment.
  """

    default_experiments = ['snp']

    def __init__(self,
                 experiment,
                 market_file,
                 additional_file=None,
                 regime_file=None,
                 root_folder=None,
                 macd_vars=None,
                 returns_vars=None,
                 add_prefix_col=None,
                 add_macd_vars=None,
                 add_returns_vars=None,
                 true_target=None,
                 ):
        """Creates configs based on default experiment chosen.

    Args:
      experiment: Name of experiment.
      root_folder: Root folder to save all outputs of training.
    """


        if experiment not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(experiment))

        # Defines all relevant paths
        if root_folder is None:
            root_folder = os.path.normpath(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '..', 'outputs'))
            print('Using root folder {}'.format(root_folder))

        in_folder = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        self.macd_vars = macd_vars
        self.returns_vars = returns_vars
        self.market_file = market_file
        self.additional_file = additional_file
        self.regime_file = regime_file
        self.root_folder = root_folder
        self.experiment = experiment
        self.add_prefix_col = add_prefix_col
        self.add_macd_vars = add_macd_vars
        self.add_returns_vars = add_returns_vars
        self.data_folder = os.path.join(root_folder, 'data', experiment)
        self.model_folder = os.path.join(root_folder, 'saved_models', experiment)
        self.results_folder = os.path.join(root_folder, 'results', experiment)
        self.split_folder = os.path.join(in_folder, 'split', 'res')
        self.regime_folder = os.path.join(in_folder, 'regime', 'res')
        self.additional_folder = os.path.join(in_folder, 'additional_data', 'res')
        self.true_target = true_target

        # Creates folders if they don't exist
        for relevant_directory in [
            self.root_folder, self.data_folder, self.model_folder,
            self.results_folder
        ]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_csv_path(self):
        csv_map = {
            'snp': 'formatted_omi_vol.csv'
        }

        return os.path.join(self.data_folder, csv_map[self.experiment])

    @property
    def data_config(self):
        add_path = None if self.additional_file is None else os.path.join(self.additional_folder,
                                                                          self.additional_file + '.z')
        reg_path = None if self.regime_folder is None else os.path.join(self.regime_folder, self.regime_file + '.z')

        ans = {
            'split_file': os.path.join(self.split_folder, self.market_file + '.z'),
            'regime_file': reg_path,
            'additional_file': add_path,
            'macd_vars': self.macd_vars,
            'returns_vars': self.returns_vars,
            'add_macd_vars': self.add_macd_vars,
            'add_returns_vars': self.add_returns_vars,
            'add_prefix_col': self.add_prefix_col,
            'true_target': self.true_target,
        }

        return ans

    @property
    def hyperparam_iterations(self):

        return 240

    def make_data_formatter(self):
        """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

        data_formatter_class = {
            'snp': SnPFormatter,
        }

        return data_formatter_class[self.experiment]()
