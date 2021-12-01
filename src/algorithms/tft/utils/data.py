from algorithms.tft.libs.tft_model import InputTypes
from algorithms.tft.libs.utils import get_single_col_by_input_type, extract_cols_from_data_type


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
    historical_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.OBSERVED_INPUT, InputTypes.STATIC_INPUT}
    ]

    historical_cols += static_cols

    col_mappings = {
        'identifier': [id_col],
        'time': [time_col],
        'outputs': [target_col],
        'inputs': input_cols,
        'historical_inputs': historical_cols
    }
    return col_mappings
