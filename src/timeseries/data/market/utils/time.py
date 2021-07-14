import pandas as pd


def insert_weekend(df, s_thold=150000):
    diff = pd.Series(df.index).diff()
    weekend = (diff.dt.total_seconds() > s_thold).astype(int)
    return df.insert(0, 'weekend', weekend.values, True)