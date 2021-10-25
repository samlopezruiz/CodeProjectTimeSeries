# Lint as: python3
"""Custom formatting functions for Volatility dataset.

Defines dataset specific column definitions and data transformations.
"""
import joblib
import pandas as pd
import sklearn.preprocessing

import src.algorithms.tft2.libs.utils as utils
from algorithms.hmm.func import resample_dfs
from algorithms.tft2.data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
from timeseries.experiments.market.preprocess.func import add_features
from timeseries.experiments.market.utils.data import new_cols_names

_add_column_definition = [
    ('NQc_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('NQ_atr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('NQc_macd', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('NQ_volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)
]


class SnPFormatter(GenericDataFormatter):
    """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

    _column_definition = [
        ('subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('datetime', DataTypes.DATE, InputTypes.TIME),
        ('ESc_r', DataTypes.REAL_VALUED, InputTypes.TARGET),
        # ('ESc', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # ('NQc', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESo_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESh_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESl_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('adl', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('delta', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('atr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESc_macd', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('days_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hour_of_day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('regime', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    _column_definition += _add_column_definition

    fixed_params = {
        'quantiles': [0.1, 0.5, 0.9],
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'multiprocessing_workers': 12,
    }

    model_params = {
        'total_time_steps': 50 + 5,
        'num_encoder_steps': 50,
        'dropout_rate': 0.3,
        'hidden_layer_size': 160,
        'learning_rate': 0.01,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1,
    }

    def __init__(self):
        """Initialises formatter."""
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.n_states = None
        self.valid_true_y = None
        self.test_true_y = None

    # def split_data(self, df, valid_boundary=2016, test_boundary=2018):
    #     pass

    def split_data(self, data_config, valid_boundary=2016, test_boundary=2018):
        """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      data_config: Source config to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """
        mkt_data, add_data, reg_data = self.load_data(data_config)

        # Add processed features
        add_features(mkt_data,
                     macds=data_config['macd_vars'],
                     returns=data_config['returns_vars'])

        if add_data is not None:
            additional_resampled = resample_dfs(mkt_data, add_data)
            additional_resampled.columns = new_cols_names(additional_resampled, data_config['add_prefix_col'])

            # append additional features
            add_features(additional_resampled,
                         macds=data_config['add_macd_vars'],
                         returns=data_config['add_returns_vars'])

            # discard all but relevant features ?
            # additional_resampled = additional_resampled.loc[:, [col[0] for col in _add_column_definition]]
            mkt_data = pd.concat([mkt_data, additional_resampled], axis=1)

        if reg_data is not None:
            reg_resampled = resample_dfs(mkt_data, reg_data)
            mkt_data['regime'] = reg_resampled['state']

        print('\n{} Available Features: {}'.format(mkt_data.shape[1], list(mkt_data.columns)))

        mkt_data['datetime'] = mkt_data.index
        mkt_data.reset_index(drop=True, inplace=True)
        train, test = mkt_data.loc[mkt_data.loc[:, 'test'] == 0, :], mkt_data.loc[mkt_data.loc[:, 'test'] == 1, :]

        if data_config['true_target'] is not None:
            self.set_true_target(data_config['true_target'], test, test)

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, test, test])

    def load_data(self, data_config):
        print('Loading Market Data: {}'.format(data_config['split_file']))
        split_data = joblib.load(data_config['split_file'])
        if data_config['additional_file']:
            print('   Loading Add Data: {}'.format(data_config['additional_file']))
            additional_data = joblib.load(data_config['additional_file'])
        else:
            additional_data = None
        if data_config['regime_file']:
            print('Loading Regime Data: {}'.format(data_config['regime_file']))
            regime_data = joblib.load(data_config['regime_file'])
            self.n_states = regime_data['n_regimes']
        else:
            regime_data = None
        mkt_data = split_data['data']
        add_data = additional_data.get('data', None)
        reg_data = regime_data.get('data', None)

        return mkt_data, add_data, reg_data

    def set_true_target(self, true_target, valid, test):

        print('Setting target vars for reconstruction...')
        column_definitions = self.get_column_definition()
        time_column = utils.get_single_col_by_input_type(InputTypes.TIME, column_definitions)
        self.valid_true_y = pd.Series(valid[true_target].values, index=valid[time_column], name=true_target).to_frame()
        self.test_true_y = pd.Series(test[true_target].values, index=valid[time_column], name=true_target).to_frame()

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()

        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    # Default params
    def update_model_params(self, new_cfg):
        for key, val in new_cfg.items():
            self.model_params[key] = val

    def update_fixed_params(self, new_cfg):
        for key, val in new_cfg.items():
            self.fixed_params[key] = val

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        return self.fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        return self.model_params
