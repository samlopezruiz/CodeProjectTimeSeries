import numpy as np
import pandas as pd

def ismv(train):
    if len(train.shape) > 1:
        is_mv = True if train.shape[1] > 1 else False
    else:
        is_mv = False
    return is_mv

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


def atr_bad(df, inst, n=14):
    high_low = df[inst + 'h'] - df[inst + 'l']
    high_close = np.abs(df[inst + 'h'] - df[inst + 'c'].shift())
    low_close = np.abs(df[inst + 'l'] - df[inst + 'c'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    atr = true_range.rolling(n).sum() / n
    return atr


def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1 / n, adjust=False).mean()


def atr(df, inst, n=14):
    data = df.copy()
    high = df[inst + 'h']
    low = df[inst + 'l']
    close = df[inst + 'c']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr


def macd(x, p0=12, p1=26):
    # x = np.array(x)
    ema0 = ema(x, p0)
    ema1 = ema(x, p1)
    return ema0 - ema1