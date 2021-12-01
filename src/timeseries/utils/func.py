import os
import pandas as pd


def rename_cols(df_x, df_y, num_inst):
    cols_x = list(df_x.columns)
    cols_y = list(df_y.columns)
    suffixes = ["a", "b", "c", "d", "e"]

    for i in range(len(cols_y)):
        if cols_y[i] in cols_x:
            cols_y[i] = cols_y[i] + "_" + suffixes[num_inst]
    return cols_x, cols_y


def sync_df(df_x, df_y, features_x=None, features_y=None, num_inst=1, index_col='index'):
    if features_y is None:
        features_y = features_x
    if features_x is None:
        df_x, df_y = df_x, df_y
    else:
        df_x, df_y = df_x.loc[:, features_x], df_y.loc[:, features_y]

    index_x, index_y = list(df_x.index), list(df_y.index)
    y, ix_y = 0, []

    for x, ix_x in enumerate(index_x):
        while index_y[y] < ix_x and y < len(index_y) - 1:
            y += 1
        ix_y.append(max(0, y - 1))

    df_y_new = df_y.iloc[ix_y, :]
    df_y_new.reset_index(inplace=True)
    df_x.reset_index(inplace=True)

    cols_x, cols_y = rename_cols(df_x, df_y_new, num_inst)

    df_synced = pd.concat([df_x, df_y_new], axis=1, ignore_index=True)
    df_synced.columns = cols_x + cols_y
    df_synced.set_index(index_col, inplace=True)
    return df_synced


def interpolate(downsample, upsample):
    new_ix = upsample.index.tolist()
    old_ix = downsample.index.tolist()
    old_x = downsample.values.tolist()
    new_x = []
    i = 1
    x0, x1 = old_x[0], old_x[1]
    t0, t1 = old_ix[0], old_ix[1]
    for t in new_ix:
        if t > t1:
            i = i + 1 if i < len(old_x) else i
            x0, x1 = old_x[i - 1], old_x[i]
            t0, t1 = old_ix[i - 1], old_ix[i]
        x = polation(x1, x0, t0, t1, t)
        new_x.append(x)
    return pd.Series(new_x, index=upsample.index)


def polation(x1, x0, t0, t1, t):
    assert t1 - t0 != 0
    m = (x1 - x0) / (t1 - t0)
    if t0 <= t < t1:
        x = x0 + m * (t - t0)
    elif t < t0:
        x = x0 - m * (t0 - t)
    else:
        x = x1 + m * (t - t1)
    return x

def append_interpolation_ensemble(df_upsample, downsample):
    upsample = df_upsample['forecast'].dropna()
    # interpolation = interpolate(downsample, upsample)

    df = pd.concat([upsample, downsample], axis=1)
    df.columns = ['upsample', 'downsample']
    df = df.interpolate(method='index', limit_direction='both')
    upsampled = df.loc[upsample.index, 'downsample']

    df_upsample['upsampled'] = upsampled
    df_upsample['ensemble'] = (df_upsample['upsampled'] + df_upsample['forecast']) / 2
    return df_upsample


def interpolate_series(upsample, downsample):
    df = pd.concat([upsample, downsample], axis=1)
    df.columns = ['upsample', 'downsample']
    df = df.interpolate(method='index', limit_direction='both')
    return df


