import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.plotly.plot import plotly_time_series
import seaborn as sns


def preprocess_x(x, detrend=None, scale=True, standard_scaler=None):
    if len(detrend) == 2:
        detrend, period = detrend  # ('ema_diff', 14)
    elif detrend == 'ema_diff':
        print('ERROR: specify period if detrend is emma_diff')
        return

    if detrend == 'ln_return':
        x = ln_returns(x)
    if detrend == 'ema_diff':
        x = np.array(x) - ema(x, period)
    if detrend == 'diff':
        x = np.diff(np.array(x), axis=0)
    if scale:
        if standard_scaler is None:
            standard_scaler = StandardScaler()
            x = standard_scaler.fit_transform(x)
        else:
            x = standard_scaler.transform(x)

    return x, standard_scaler


def reconstruct_x(a_1, forecast, feature_col, steps=1, test=None, standscaler=None, detrend='ln_return'):
    if len(detrend) == 2:
        detrend, period = detrend  # ('ema_diff', 14)
    elif detrend == 'ema_diff':
        print('ERROR: specify period if detrend is emma_diff')
        return

    # forecast = prep_forecast(forecast)
    forecast = inverse_scaler(forecast, standscaler)

    forecast = forecast[:, feature_col]
    if test is not None:
        test = test[:, feature_col]
    if detrend == 'ln_return':
        pred = reconstruct_from_ln_r(forecast, a_1, steps, test)
    elif detrend == 'ema_diff':
        pred = reconstruct_from_ema_diff(forecast, a_1, steps, test, period)
    elif detrend == 'diff':
        pred = reconstruct_from_diff(forecast, a_1, steps, test)
    else:
        pred = forecast
    return np.array(pred)


def ln_returns(x):
    ln_r = np.log(x) - np.log(np.roll(x, 1, axis=0))
    # first row has no returns
    return ln_r[1:]


def ema(x, period, last_ema=None):
    c1 = 2 / (1 + period)
    c2 = 1 - (2 / (1 + period))
    x = np.array(x)
    if last_ema is None:
        ema_x = np.array(x)
        for i in range(1, ema_x.shape[0]):
            ema_x[i] = x[i] * c1 + c2 * ema_x[i - 1]
    else:
        ema_x = np.zeros((len(x) + 1,))
        ema_x[0] = last_ema
        for i in range(1, ema_x.shape[0]):
            ema_x[i] = x[i] * c1 + c2 * ema_x[i - 1]
        ema_x = ema_x[1:]
    return ema_x


def series_from_ln_r(p_1, ln_r):
    p = []
    for r in ln_r:
        p_1 = np.exp(np.log(p_1) + r)
        p.append(p_1)
    return np.array(p)


def series_from_diff(p_1, diff):
    p = []
    for d in diff:
        p_1 = p_1 + d
        p.append(p_1)
    return np.array(p)


def series_from_ema_diff(ema_1, ema_diff, period):
    p = []
    c1 = 2 / (1 + period)
    c2 = 1 - (2 / (1 + period))
    ema_x = np.zeros((len(ema_diff)+1,))
    ema_x[0] = ema_1
    for i, e_diff in enumerate(ema_diff):
        ema_x[i+1] = (ema_x[i] + e_diff) * c1 + c2 * ema_x[i]
        p.append(e_diff + ema_x[i+1])

    return np.array(p)


def reconstruct_from_diff(diff, p_1, steps, test):
    if test is not None:
        pred = []
        actual = np.hstack((p_1, test))  # actual data starts with p(t-1)
        for i in range(0, len(test), steps):
            end = min(i + steps, len(diff))
            # returns have a 1 delay, therefore P(t) = exp(r(t)+P(t-1))
            y_pred = series_from_diff(actual[i], diff[i:end])
            [pred.append(y) for y in y_pred]
    else:
        pred = series_from_diff(p_1, diff)
        pred = np.hstack((p_1, pred))
    return pred


def reconstruct_from_ema_diff(ema_diff, ema_1, steps, test, period):
    if test is not None:
        pred = []
        ema_ref = ema(test, period, last_ema=ema_1)

        for i in range(0, len(test), steps):
            end = min(i + steps, len(ema_diff))
            y_pred = series_from_ema_diff(ema_ref[i], ema_diff[i:end], period)
            [pred.append(y) for y in y_pred]
    else:
        pred = series_from_ema_diff(ema_1, ema_diff, period)
    return pred


def reconstruct_from_ln_r(ln_r, p_1, steps, test):
    if test is not None:
        pred = []
        actual = np.hstack((p_1, test))  # actual data starts with p(t-1)
        for i in range(0, len(test), steps):
            end = min(i + steps, len(ln_r))
            # returns have a 1 delay, therefore P(t) = exp(r(t)+P(t-1))
            y_pred = series_from_ln_r(actual[i], ln_r[i:end])
            [pred.append(y) for y in y_pred]
    else:
        pred = series_from_ln_r(p_1, ln_r)
        pred = np.hstack((p_1, pred))
    return pred


def inverse_scaler(ln_r, standscaler):
    if standscaler is not None:
        s = ln_r.shape[1] if len(ln_r.shape) > 1 else 1
        if standscaler.n_features_in_ > s:
            n = standscaler.n_features_in_ - s + 1
            ln_r = standscaler.inverse_transform(np.array([ln_r] * n).T)
        else:
            ln_r = standscaler.inverse_transform(ln_r)
    return ln_r


def preprocess(input_cfg, train, test):
    if input_cfg.get('preprocess', False):
        detrend = input_cfg.get('detrend', 'ln_return')
        train_pp, ss = preprocess_x(train, detrend=detrend)
        # remove only 1 element at beginning
        all_pp, _ = preprocess_x(np.vstack((train, test)), detrend=detrend, standard_scaler=ss)
        test_pp = all_pp[train_pp.shape[0]:]
        # plotly_time_series(pd.DataFrame(train_pp), rows=[0, 1, 2, 3], markers='lines')
    else:
        ss = None
        train_pp, test_pp = train, test
    return train_pp, test_pp, ss


def reconstruct(forecast, train, test, input_cfg, cfg, ss=None):
    if input_cfg.get('preprocess', False):
        return reconstruct_x(train[-1, -1], forecast, train.shape[1] - 1,
                             detrend=input_cfg.get('detrend', 'ln_return'),
                             standscaler=ss, test=test, steps=cfg.get('n_steps_out',1))
    else:
        return forecast


if __name__ == '__main__':
    # %% INPUT
    save_folder = 'images'
    plot_title = True
    save_plots = True
    name = "CONV-LSTM"
    detrend_ops = ['ln_return', ('ema_diff', 14), 'diff']
    input_cfg = {"variate": "multi", "granularity": 5, "noise": False, "positive_offset": True,
                 'detrend': detrend_ops[2]}
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)

    # %% RETURNS
    train_pp, ss = preprocess_x(train, detrend=input_cfg['detrend'], scale=True)
    plotly_time_series(pd.DataFrame(train_pp), rows=[0, 1, 2, 3], markers='lines')

    # %% RECONSTRUCT ORIGINAL SERIES
    feature_col = 3
    train_reconst = reconstruct_x(train[0, -1], train_pp, feature_col,
                                  standscaler=ss, detrend=input_cfg['detrend'])


    df = pd.DataFrame([train[:, feature_col], train_reconst]).T
    df.columns = ['p', 'p_reconst']
    plotly_time_series(df, markers='lines')


