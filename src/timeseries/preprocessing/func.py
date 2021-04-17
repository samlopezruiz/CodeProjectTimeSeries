import numpy as np


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