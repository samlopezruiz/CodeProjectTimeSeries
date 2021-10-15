def new_cols_names(df, new_prefix=None):
    new_cols = list(df.columns)
    for i, col in enumerate(df.columns):
        if col[:2] != new_prefix:
            new_cols[i] = new_prefix + '_' + col
    return new_cols