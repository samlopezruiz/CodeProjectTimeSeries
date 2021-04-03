import numpy as np
import pandas as pd


def one_step_forecast_df(train, test, pred, t_train=None, t_test=None, train_prev_steps=None):
    df, t_test = prep_df(train_prev_steps, t_test, t_train, test, train)
    df_forecast = pd.DataFrame([test[0], pred])
    df_forecast = df_forecast.T
    df_forecast.index, df_forecast.columns = [t_test[0]], ['data', 'forecast']
    return pd.concat([df, df_forecast], axis=0)


def multi_step_forecast_df(train, test, pred, t_train=None, t_test=None, train_prev_steps=None):
    df, t_test = prep_df(train_prev_steps, t_test, t_train, test, train)
    df_forecast = pd.DataFrame([test, pred])
    df_forecast = df_forecast.T
    df_forecast.index, df_forecast.columns = t_test, ['data', 'forecast']
    return pd.concat([df, df_forecast], axis=0)


def prep_df(train_prev_steps, t_test, t_train, test, train):
    if train_prev_steps is None: train_prev_steps = len(train)
    if t_train is None: t_train = list(range(len(train)))
    if t_test is None: t_test = np.array(range(len(test))) + len(train)
    df = pd.DataFrame(data=train[-train_prev_steps:], index=t_train[-train_prev_steps:], columns=['train'])
    df['data'] = np.nan
    df['forecast'] = np.nan
    return df, t_test
