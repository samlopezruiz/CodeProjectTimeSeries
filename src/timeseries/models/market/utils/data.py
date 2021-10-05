def new_cols_names(df, inst0):
    new_cols = list(df.columns)
    for i, col in enumerate(df.columns):
        if col[:2] != inst0:
            new_cols[i] = inst0 + '_' + col
    return new_cols