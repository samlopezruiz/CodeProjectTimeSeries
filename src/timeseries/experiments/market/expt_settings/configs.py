import os

from timeseries.data.market.utils.names import get_inst_ohlc_names
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

        self.architecture = cfg.get('architecture', 'TFTModel')
        self.experiment = formatter
        self.root_folder = root_folder
        self.dataset_config = cfg.get('dataset_config', None)
        self.vars_definition = cfg['vars_definition']
        self.data_folder = os.path.join(root_folder, 'data', formatter)
        self.model_folder = os.path.join(root_folder, 'saved_models', formatter)
        self.results_folder = os.path.join(root_folder, 'results', formatter)
        self.split_folder = os.path.join(in_folder, 'split', 'res')
        self.regime_folder = os.path.join(in_folder, 'regime', 'res')
        self.additional_folder = os.path.join(in_folder, 'additional_data', 'res')

        dataset_cfg = self.get_dataset_cfg()

        # dataset configurations of timeseries
        self.macd_periods = dataset_cfg.get('macd_periods', [])
        self.macd_vars = dataset_cfg.get('macd_vars', [])
        self.rsi_vars = dataset_cfg.get('rsi_vars', [])
        self.returns_vars = dataset_cfg.get('returns_vars', [])
        self.market_file = dataset_cfg['market_file']
        self.additional_file = dataset_cfg.get('additional_file', None)
        self.regime_file = dataset_cfg.get('regime_file', None)
        self.add_prefix_col = dataset_cfg.get('additional_prefix_col', None)
        self.add_macd_vars = dataset_cfg.get('additional_macd_vars', None)
        self.add_returns_vars = dataset_cfg.get('additional_returns_vars', None)
        self.true_target = dataset_cfg.get('true_target', None)

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
            'rsi_vars': self.rsi_vars,
            'macd_periods': self.macd_periods,
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

        return data_formatter_class[self.experiment](self.vars_definition, self.architecture)

    def get_dataset_cfg(self):
        dataset_class = {}

        dataset_class['ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX'] = \
            {'market_file': 'split_ES_minute_60T_dwn_smpl_2018-01_to_2021-06_g12week_r25_4',
             'additional_file': 'subset_NQ_minute_60T_dwn_smpl_2012-01_to_2021-07',
             'regime_file': 'regime_ESc_r_ESc_macd_T10Y2Y_VIX',
             'macd_vars': ['ESc'],
             'rsi_vars': ['ESc'],
             'macd_periods': [12, 6],
             'returns_vars': get_inst_ohlc_names('ES'),
             'additional_prefix_col': 'NQ',
             'additional_macd_vars': ['NQc'],
             'additional_returns_vars': get_inst_ohlc_names('NQ'),
             'true_target': 'ESc'}

        dataset_class['ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2015-01_to_2021-06'] = \
            {'market_file': 'split_ES_minute_60T_dwn_smpl_2015-01_to_2021-06_g12week_r25_1',
             'additional_file': 'subset_NQ_minute_60T_dwn_smpl_2012-01_to_2021-07',
             'regime_file': 'regime_ESc_r_ESc_macd_T10Y2Y_VIX',
             'macd_vars': ['ESc'],
             'rsi_vars': ['ESc'],
             'macd_periods': [12, 6],
             'returns_vars': get_inst_ohlc_names('ES'),
             'additional_prefix_col': 'NQ',
             'additional_macd_vars': ['NQc'],
             'additional_returns_vars': get_inst_ohlc_names('NQ'),
             'true_target': 'ESc'}


        dataset_class['ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2015-01_to_2021-06_macd'] = \
            {'market_file': 'split_ES_minute_60T_dwn_smpl_2015-01_to_2021-06_g12week_r25_1',
             'additional_file': 'subset_NQ_minute_60T_dwn_smpl_2012-01_to_2021-07',
             'regime_file': 'regime_ESc_r_ESc_macd_T10Y2Y_VIX',
             'macd_vars': ['ESc'],
             'rsi_vars': ['ESc'],
             'macd_periods': [12, 6],
             'returns_vars': get_inst_ohlc_names('ES'),
             'additional_prefix_col': 'NQ',
             'additional_macd_vars': ['NQc'],
             'additional_returns_vars': get_inst_ohlc_names('NQ'),
             'true_target': None}

        dataset_class['ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2018-01_to_2021-06'] = \
            dataset_class['ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2015-01_to_2021-06']

        dataset_class['ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2018-01_to_2021-06']['market_file'] = \
            'split_ES_minute_60T_dwn_smpl_2018-01_to_2021-06_g12week_r25_6'


        if self.dataset_config not in dataset_class:
            raise Exception('{} not found in dataset configurations. '
                            '\nOptions are: \n{}'.format(self.dataset_config, dataset_class.keys()))

        return dataset_class[self.dataset_config]

