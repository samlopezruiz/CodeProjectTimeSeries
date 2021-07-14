import copy

import numpy as np
import pandas as pd

from timeseries.utils.dataframes import trim_min_len


def one_step_forecast_df(train, test, pred, t_train=None, t_test=None, train_prev_steps=None):
    df, t_test = prep_df(train_prev_steps, t_test, t_train, test, train)
    df_forecast = pd.DataFrame([test[0], pred])
    df_forecast = df_forecast.T
    df_forecast.index, df_forecast.columns = [t_test[0]], ['data', 'forecast']
    return pd.concat([df, df_forecast], axis=0)


def multi_step_forecast_df(train, test, pred, t_train=None, t_test=None, reg_prob_train=None, reg_prob_test=None,
                           train_prev_steps=None):
    df, t_test = prep_df(train_prev_steps, t_test, t_train, test, train)
    df_forecast = pd.DataFrame([test, pred])
    df_forecast = df_forecast.T
    df_forecast.index, df_forecast.columns = t_test, ['data', 'forecast']
    return pd.concat([df, df_forecast], axis=0)


def merge_forecast_df(test, pred, t_test=None, reg_prob=None):
    assert len(test) == len(pred)
    _, t_test = trim_min_len(test, t_test)
    if reg_prob is None:
        df = pd.DataFrame(np.transpose([test, pred]), index=t_test, columns=['data', 'forecast'])
    else:
        data = np.hstack([test.reshape(-1, 1), pred.reshape(-1, 1), reg_prob])
        columns = ['data', 'forecast'] + ['regime ' + str(i) for i in range(reg_prob.shape[1])]
        df = pd.DataFrame(data, index=t_test, columns=columns)
    return df


def multi_step_forecasts_df(train, test, names, forecasts, t_train=None, t_test=None, train_prev_steps=None):
    df, t_test = prep_df(train_prev_steps, t_test, t_train, test, train, n_forecasts=len(names))
    preds = copy.copy(forecasts)
    preds.insert(0, test)
    df_forecast = pd.DataFrame(np.array(preds)).T
    df_forecast.index = t_test
    df_forecast.columns = ['data'] + ['forecast' + str(i) for i in range(len(names))]
    df_result = pd.concat([df, df_forecast], axis=0, ignore_index=True)
    df_result.columns = ['train', 'data'] + names
    return df_result


def prep_df(train_prev_steps, t_test, t_train, test, train, n_forecasts=1):
    if train_prev_steps is None: train_prev_steps = len(train)
    if t_train is None: t_train = list(range(len(train)))
    if t_test is None: t_test = np.array(range(len(test))) + len(train)
    df = pd.DataFrame(data=train[-train_prev_steps:], index=t_train[-train_prev_steps:], columns=['train'])
    df['data'] = np.nan
    if n_forecasts == 1:
        df['forecast'] = np.nan
    else:
        for i in range(n_forecasts):
            df['forecast'+str(i)] = np.nan
    return df, t_test
