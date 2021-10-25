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
                 formatter,
                 cfg,
                 ):
        """Creates configs based on default experiment chosen.

    Args:
      experiment: Name of experiment.
      root_folder: Root folder to save all outputs of training.
    """


        if formatter not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(formatter))

        root_folder = cfg.get('root_folder', None)

        # Defines all relevant paths
        if root_folder is None:
            root_folder = os.path.normpath(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '..', 'outputs'))
            print('Using root folder {}'.format(root_folder))

        in_folder = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        self.macd_vars = cfg.get('macd_vars', [])
        self.returns_vars = cfg.get('returns_vars', [])
        self.market_file = cfg['market_file']
        self.additional_file = cfg.get('additional_file', None)
        self.regime_file = cfg.get('regime_file', None)
        self.root_folder = root_folder
        self.experiment = formatter
        self.add_prefix_col = cfg.get('add_prefix_col', None)
        self.add_macd_vars = cfg.get('add_macd_vars', None)
        self.add_returns_vars = cfg.get('add_returns_vars', None)
        self.data_folder = os.path.join(root_folder, 'data', formatter)
        self.model_folder = os.path.join(root_folder, 'saved_models', formatter)
        self.results_folder = os.path.join(root_folder, 'results', formatter)
        self.split_folder = os.path.join(in_folder, 'split', 'res')
        self.regime_folder = os.path.join(in_folder, 'regime', 'res')
        self.additional_folder = os.path.join(in_folder, 'additional_data', 'res')
        self.true_target = cfg.get('true_target', None)

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
