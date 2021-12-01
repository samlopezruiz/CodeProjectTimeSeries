from algorithms.tft.libs.tft_model import InputTypes
from algorithms.tft.libs.utils import get_single_col_by_input_type, extract_cols_from_data_type
from timeseries.experiments.market.data_formatter.base import DataTypes


def get_col_mapping(column_definition):
    id_col = get_single_col_by_input_type(InputTypes.ID, column_definition)
    time_col = get_single_col_by_input_type(InputTypes.TIME, column_definition)
    target_col = get_single_col_by_input_type(InputTypes.TARGET, column_definition)
    static_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] == InputTypes.STATIC_INPUT
    ]
    input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]
    unkown = [
        tup[0]
        for tup in column_definition
        if tup[2] == InputTypes.OBSERVED_INPUT
    ]
    kown = [
        tup[0]
        for tup in column_definition
        if tup[2] == InputTypes.KNOWN_INPUT
    ]
    obs = [
        tup[0]
        for tup in column_definition
        if tup[2] == InputTypes.TARGET
    ]
    # historical_cols = [
    #     tup[0]
    #     for tup in column_definition
    #     if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.STATIC_INPUT}
    # ]
    historical_cols = unkown + kown + obs


    col_mappings = {
        'identifier': [id_col],
        'time': [time_col],
        'outputs': [target_col],
        'inputs': input_cols,
        'historical_inputs': historical_cols
    }
    return col_mappings


def extract_numerical_data(data):
    """Strips out forecast time and identifier columns."""
    return data[[
        col for col in data.columns
        if col not in {"forecast_time", "identifier"}
    ]]
