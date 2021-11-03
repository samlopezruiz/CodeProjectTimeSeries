from timeseries.experiments.market.data_formatter.base import DataTypes, InputTypes

variable_definitions = {}

variable_definitions['ES_r_all'] = \
    [
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

        # additional vars
        ('NQc_r', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('NQ_atr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('NQc_macd', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('NQ_volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)
    ]
