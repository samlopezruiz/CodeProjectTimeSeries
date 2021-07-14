import numpy as np
from sklearn.preprocessing import StandardScaler

from timeseries.preprocessing.func import ln_returns, ema, ismv, macd


def preprocess_x(x, detrend=None, scale=True, standard_scaler=None, ema_period=0):
    if len(detrend) == 2:
        detrend, period = detrend  # ('ema_diff', 14)
    elif detrend == 'ema_diff':
        print('ERROR: specify period if detrend is emma_diff')
        return

    if detrend == 'ln_return':
        if ema_period > 0:
            x = ln_returns(ema(x, ema_period))
        else:
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


def reconstruct_x(forecast, y_col, steps=1, test=None, standscaler=None, a_1=None, detrend='ln_return'):
    if a_1 is None and test is None:
        raise Exception('test or a_1 have to be specified')
    elif test is not None:
        a_1 = test[0]
        test_ = test[1:]
    if len(detrend) == 2:
        detrend, period = detrend  # ('ema_diff', 14)
    elif detrend == 'ema_diff':
        print('ERROR: specify period if detrend is emma_diff')
        return

    all_var_unscaled = inverse_scaler(forecast, standscaler)
    if ismv(all_var_unscaled):
        forecast_unscaled = all_var_unscaled[:, y_col]
    else:
        forecast_unscaled = all_var_unscaled

    if detrend == 'ln_return':
        pred = reconstruct_from_ln_r(forecast_unscaled, a_1, steps, test_)
    elif detrend == 'ema_diff':
        pred = reconstruct_from_ema_diff(forecast_unscaled, a_1, steps, test_, period)
    elif detrend == 'diff':
        pred = reconstruct_from_diff(forecast_unscaled, a_1, steps, test_)
    else:
        pred = forecast_unscaled
    return np.array(pred).astype(float).reshape(-1, )


def series_from_ln_r(p_1, ln_r):
    p = []
    if isinstance(p_1, np.ndarray):
        p_1 = p_1[0]
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
    ema_x = np.zeros((len(ema_diff) + 1,))
    ema_x[0] = ema_1
    for i, e_diff in enumerate(ema_diff):
        ema_x[i + 1] = (ema_x[i] + e_diff) * c1 + c2 * ema_x[i]
        p.append(e_diff + ema_x[i + 1])

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
        test = test.reshape(-1, 1)
        pred = [p_1]
        actual = np.vstack((p_1, test))  # actual data starts with p(t-1)
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


def preprocess(input_cfg, train, test, reg_prob_train=None, reg_prob_test=None, ss=None):
    if input_cfg.get('preprocess', False):
        detrend = input_cfg.get('detrend', 'ln_return')
        if len(train.shape) == 1:
            train = train.reshape(-1, 1)
            test = test.reshape(-1, 1)
        ema_period = input_cfg.get('ema_period', 0)
        train_pp, ss = preprocess_x(train, detrend=detrend, ema_period=ema_period, standard_scaler=ss)
        test_pp, _ = preprocess_x(test, detrend=detrend, ema_period=ema_period, standard_scaler=ss)

        reg_prob_train = reg_prob_train[1:] if reg_prob_train is not None else None
        reg_prob_test = reg_prob_test[1:] if reg_prob_test is not None else None
        return train_pp, test_pp, reg_prob_train, reg_prob_test, ss
    else:
        return train, test, reg_prob_train, reg_prob_test, ss


def reconstruct_pred(forecast, input_cfg, model_n_steps_out, test=None, ss=None, a_1=None):
    if input_cfg.get('preprocess', False):
        assert len(test) == len(forecast) + 1
        if len(test.shape) == 1:
            test = test.reshape(-1, 1)
        y_col = ss.n_features_in_ - 1 if ss is not None else None
        return reconstruct_x(forecast, y_col, a_1=a_1,
                             detrend=input_cfg.get('detrend', 'ln_return'),
                             standscaler=ss, test=test, steps=model_n_steps_out)
    else:
        return forecast


def add_features(df, macds=None, returns=None):
    if returns is not None:
        for var in returns:
            df[var+'_r'] = ln_returns(df[var])
    if macds is not None:
        for var in macds:
            df[var+'_macd'] = macd(df[var])
    df.dropna(inplace=True)