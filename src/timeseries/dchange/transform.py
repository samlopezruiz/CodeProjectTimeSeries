from timeseries.data.lorenz.lorenz import multivariate_lorenz
from timeseries.plotly.plot import plotly_time_series, plotly_one_series
import numpy as np
import pandas as pd


def direct_change(s, thold=0.02):
    ix = np.array(s.index)
    p_h, p_l = (s[0], s[0])
    t0_dc, t1_dc, t0_os, t1_os = (ix[0], ix[0], ix[0], ix[0])
    upturn = True
    dc_events = pd.Series(dtype=np.float64, name='dc')
    tmv_events = pd.Series(dtype=np.float64, name='tmv')
    r_events = pd.Series(dtype=np.float64, name='r')
    t_events = pd.Series(dtype=np.float64, name='t')
    for t, p in s.items():
        if upturn:
            if p <= p_h * (1 - thold):
                upturn = False
                # end time for a downturn event
                t1_dc = t
                # start time for a downward OS (t+1)
                t0_os = t
                dc_events = dc_events.append(pd.Series([p_h, p], index=[t0_dc, t], name='dc'))
                if dc_events.shape[0] >= 4:
                    tmv_events = tmv_events.append(
                        pd.Series([(p_h - p_l) / (thold * p_l)], index=[t0_dc], name='tmv'))
                    t_events = t_events.append(pd.Series([dc_events.index[-2] - dc_events.index[-4]],
                                                         index=[t0_dc], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc], name='r'))
                p_l = p
                # start time for a upturn event
                t0_dc = t
                # end time for a downward OS (t-1)
                t1_os = t
            elif p_h < p:
                p_h = p
                # start time for a downturn event
                t0_dc = t
                # end time for a upward OS (t-1)
                t1_os = t
        else:
            if p >= p_l * (1 + thold):
                upturn = True
                # end time for a upturn event
                t1_dc = t
                # start time for a upturn OS (t+1)
                t0_os = t
                dc_events = dc_events.append(pd.Series([p_l, p], index=[t0_dc, t], name='dc'))
                if dc_events.shape[0] >= 4:
                    tmv_events = tmv_events.append(
                        pd.Series([(p_h - p_l) / (thold * p_h)], index=[t0_dc], name='tmv'))
                    t_events = t_events.append(pd.Series([dc_events.index[-2] - dc_events.index[-4]],
                                                         index=[t0_dc], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc], name='r'))
                p_h = p
                # start time for a downturn event
                t0_dc = t
                # end time for a upward OS (t-1)
                t1_os = t
            elif p_l > p:
                p_l = p
                # start time for a upturn event
                t0_dc = t
                # end time for a downward OS (t-1)
                t1_os = t
    print("duplicated = {}".format(sum(dc_events.index.duplicated(keep="first"))))
    dc_events = dc_events[~dc_events.index.duplicated(keep="first")]
    return pd.DataFrame([dc_events, tmv_events, r_events, t_events]).T


if __name__ == '__main__':
    # %% INPUTS
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=1, positive_offset=True)
    name = "DIRECTIONAL CHANGE"
    x = lorenz_df['x']

    # %% TRANSFORM
    dc_df = direct_change(x, thold=0.02)
    dc = dc_df['dc']
    comb = pd.DataFrame([x, dc]).T
    comb.columns = [x.name, 'dc']
    plotly_time_series(comb, rows=[0, 1], title=name, markers='lines+markers')
    plotly_time_series(dc_df, rows=[0, 1, 2, 3], title=name, markers='lines+markers')

