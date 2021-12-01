import pandas as pd


def new_cols_names(df, new_prefix=None):
    new_cols = list(df.columns)
    for i, col in enumerate(df.columns):
        if col[:2] != new_prefix:
            new_cols[i] = new_prefix + '_' + col
    return new_cols

def get_column_def_df(col_def):
    col_def_df = pd.DataFrame()
    col_def_df['feature'] = [cd[0] for cd in col_def]
    col_def_df['data type'] = [cd[1].name for cd in col_def]
    col_def_df['input type'] = [cd[2].name for cd in col_def]
    col_def_df.iloc[3:, :] = col_def_df.iloc[3:, :].sort_values(by=['input type', 'data type'])
    return col_def_df