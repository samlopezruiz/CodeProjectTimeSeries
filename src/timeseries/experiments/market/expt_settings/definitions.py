import copy

from timeseries.experiments.market.data_formatter.base import DataTypes, InputTypes


def append_vars(definition, new_vars):
    definition += new_vars


variable_definitions = {}

mkt_vars = [('ESc_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('ESo_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('ESh_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('ESl_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('ESc_rsi', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('adl', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('delta', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('atr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)]

cat_date_vars = [('days_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                 ('hour_of_day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
                 ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
                 ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
                 ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
                 ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT)]

additional_market_vars = [('NQc_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                          ('NQ_atr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                          ('NQc_macd_12_26', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                          ('NQ_volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)]

vol_profile_vars = [('p_dist_-5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_-4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_-3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_-2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_-1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('p_dist_5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

                    ('norm_vol_-5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_-4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_-3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_-2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_-1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
                    ('norm_vol_5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

                    ('is_max_-5', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_-4', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_-3', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_-2', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_-1', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_1', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_2', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_3', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_4', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
                    ('is_max_5', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT)]

variable_definitions['ES_r'] = \
    [
        ('ESc_r', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('test_train_subset', DataTypes.CATEGORICAL, InputTypes.ID),
        # ('subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('datetime', DataTypes.DATE, InputTypes.TIME),
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('regime', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

append_vars(variable_definitions['ES_r'], mkt_vars)
append_vars(variable_definitions['ES_r'], cat_date_vars)
append_vars(variable_definitions['ES_r'], additional_market_vars)
append_vars(variable_definitions['ES_r'], vol_profile_vars)

variable_definitions['ES_r_vol'] = \
    [
        ('ESc_r', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('test_train_subset', DataTypes.CATEGORICAL, InputTypes.ID),
        # ('subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('datetime', DataTypes.DATE, InputTypes.TIME),
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('regime', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

append_vars(variable_definitions['ES_r_vol'], cat_date_vars)
append_vars(variable_definitions['ES_r_vol'], additional_market_vars)
append_vars(variable_definitions['ES_r_vol'], vol_profile_vars)

variable_definitions['ES_macd_vol'] = \
    [
        # TARGET
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.TARGET),
        # ('test_train_subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('datetime', DataTypes.DATE, InputTypes.TIME),
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('regime', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

append_vars(variable_definitions['ES_macd_vol'], mkt_vars)
append_vars(variable_definitions['ES_macd_vol'], cat_date_vars)
append_vars(variable_definitions['ES_macd_vol'], additional_market_vars)
append_vars(variable_definitions['ES_macd_vol'], vol_profile_vars)

variable_definitions['ES_macd'] = \
    [
        # TARGET
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.TARGET),
        # VARS
        # ('Symbol', DataTypes.CATEGORICAL, InputTypes.ID),
        ('test_train_subset', DataTypes.CATEGORICAL, InputTypes.ID),
        # ('subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('datetime', DataTypes.DATE, InputTypes.TIME),
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('regime', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

append_vars(variable_definitions['ES_macd'], mkt_vars)
append_vars(variable_definitions['ES_macd'], cat_date_vars)
append_vars(variable_definitions['ES_macd'], additional_market_vars)

variable_definitions['ES_fast_macd'] = \
    [
        # TARGET
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.TARGET),

        ('test_train_subset', DataTypes.CATEGORICAL, InputTypes.ID),
        # ('subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('datetime', DataTypes.DATE, InputTypes.TIME),
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('regime', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

append_vars(variable_definitions['ES_fast_macd'], mkt_vars)
append_vars(variable_definitions['ES_fast_macd'], cat_date_vars)
append_vars(variable_definitions['ES_fast_macd'], additional_market_vars)

variable_definitions['ES_fast_macd_vol'] = \
    [
        # TARGET
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.TARGET),

        ('test_train_subset', DataTypes.CATEGORICAL, InputTypes.ID),
        # ('subset', DataTypes.CATEGORICAL, InputTypes.ID),
        ('datetime', DataTypes.DATE, InputTypes.TIME),
        ('ESc_macd_6_12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ESc_macd_12_24', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('regime', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

append_vars(variable_definitions['ES_fast_macd_vol'], mkt_vars)
append_vars(variable_definitions['ES_fast_macd_vol'], cat_date_vars)
append_vars(variable_definitions['ES_fast_macd_vol'], additional_market_vars)
append_vars(variable_definitions['ES_fast_macd_vol'], vol_profile_vars)
