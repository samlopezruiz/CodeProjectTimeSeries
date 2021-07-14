import numpy as np
import pandas as pd
import decimal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from algorithms.hmm.func import count_regimes, get_regimes
from timeseries.preprocessing.func import ema
from datetime import timedelta

from timeseries.utils.dataframes import relabel


def direct_change(s, cfg):
    thold, delta_t, delta_y = cfg.get('thold', 0.02), cfg['delta_t'], cfg['delta_y']
    delta = delta_t / 10
    round_p = max(-decimal.Decimal(str(delta)).as_tuple().exponent, 0)
    ix = np.array(s.index)
    p_h, p_l = (s.iloc[0], s.iloc[0])
    t0_dc, t1_dc, t0_os, t1_os = (ix[0], ix[0], ix[0], ix[0])
    upturn = True
    dc_events = pd.Series(dtype=np.float64, name='dc')
    tmv_events = pd.Series(dtype=np.float64, name='tmv')
    r_events = pd.Series(dtype=np.float64, name='r')
    t_events = pd.Series(dtype=np.float64, name='t')
    t_1, p_1 = (ix[0], s.iloc[0])
    for t, p in s.items():
        thold = delta_y / p
        if upturn:
            if p <= p_h * (1 - thold):
                upturn = False
                # end time for a downturn event
                t1_dc = t
                # start time for a downward OS (t+1)
                # t0_os = t + delta_t
                t0_dc_ = round(
                    (t0_dc if t0_dc != dc_events.index[-1] else t0_dc + delta) if len(dc_events) > 0 else t0_dc,
                    round_p)
                dc_events = dc_events.append(pd.Series([p_h, p], index=[t0_dc_, t], name='dc'))
                if dc_events.shape[0] >= 4:
                    tmv_events = tmv_events.append(
                        pd.Series([(p_h - p_l) / (thold * p_l)], index=[t0_dc_], name='tmv'))
                    t_events = t_events.append(pd.Series([dc_events.index[-2] - dc_events.index[-4]],
                                                         index=[t0_dc_], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc_], name='r'))
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
                t0_dc_ = round(
                    (t0_dc if t0_dc != dc_events.index[-1] else t0_dc + delta) if len(dc_events) > 0 else t0_dc,
                    round_p)
                dc_events = dc_events.append(pd.Series([p_l, p], index=[t0_dc_, t], name='dc'))
                if dc_events.shape[0] >= 4:
                    tmv_events = tmv_events.append(
                        pd.Series([(p_h - p_l) / (thold * p_h)], index=[t0_dc_], name='tmv'))
                    t_events = t_events.append(pd.Series([dc_events.index[-2] - dc_events.index[-4]],
                                                         index=[t0_dc_], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc_], name='r'))
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
        t_1, p_1 = t, p
    duplicated_mask = dc_events.index.duplicated(keep="first")
    print("duplicated = {}".format(sum(duplicated_mask)))
    dc_events = dc_events[~duplicated_mask]
    return pd.DataFrame([dc_events, tmv_events, r_events, t_events]).T


def direct_change_ts(s, cfg, weekend=None):
    thold, delta, delta_y = cfg.get('thold', 0.02), cfg['delta_t'], cfg['delta_y']
    ix = np.array(s.index)
    p_h, p_l = (s.iloc[0], s.iloc[0])
    t0_dc, t1_dc, t0_os, t1_os = (ix[0], ix[0], ix[0], ix[0])
    upturn = True
    dc_events = pd.Series(dtype=np.float64, name='dc')
    tmv_events = pd.Series(dtype=np.float64, name='tmv')
    r_events = pd.Series(dtype=np.float64, name='r')
    t_events = pd.Series(dtype=np.float64, name='t')
    w_events = pd.Series(dtype=np.float64, name='w')
    t_1, p_1 = (ix[0], s.iloc[0])
    wd0 = False  # weekend detected, but has to wait
    wd = False
    for t, p in s.items():
        thold = delta_y / p
        if weekend is not None:
            if weekend[t] == 1 and wd is False:
                wd0 = True
        if upturn:
            if p <= p_h * (1 - thold):
                upturn = False
                # end time for a downturn event
                t1_dc = t
                # start time for a downward OS (t+1)
                t0_dc_ = (t0_dc if t0_dc != dc_events.index[-1] else t0_dc + timedelta(seconds=delta)) if len(
                    dc_events) > 0 else t0_dc
                dc_events = dc_events.append(pd.Series([p_h, p], index=[t0_dc_, t], name='dc'))
                if dc_events.shape[0] >= 4:
                    dt = (dc_events.index[-2] - dc_events.index[-4]).total_seconds()
                    tmv = (p_h - p_l) / (thold * p_l)
                    tmv_events = tmv_events.append(
                        pd.Series([tmv if not wd or not wd0 else delta_y / 2], index=[t0_dc_], name='tmv'))
                    t_fill = dt if np.mean(t_events[-2:]) is np.nan else np.mean(t_events[-2:])
                    t_events = t_events.append(pd.Series([dt if not wd and not wd0 else t_fill],
                                                         index=[t0_dc_], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc_], name='r'))
                    w_events = w_events.append(pd.Series([int(wd) + int(wd0)], index=[t0_dc_], name='w'))
                    if wd:
                        wd0 = False
                        wd = False
                    if wd0: wd = True

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
                t0_dc_ = (t0_dc if t0_dc != dc_events.index[-1] else t0_dc + timedelta(seconds=delta)) if len(
                    dc_events) > 0 else t0_dc
                dc_events = dc_events.append(pd.Series([p_l, p], index=[t0_dc_, t], name='dc'))
                if dc_events.shape[0] >= 4:
                    dt = (dc_events.index[-2] - dc_events.index[-4]).total_seconds()
                    tmv = (p_h - p_l) / (thold * p_h)
                    tmv_events = tmv_events.append(
                        pd.Series([tmv if not wd or not wd0 else delta_y / 2], index=[t0_dc_], name='tmv'))
                    t_fill = dt if np.mean(t_events[-2:]) is np.nan else np.mean(t_events[-2:])
                    t_events = t_events.append(pd.Series([dt if not wd and not wd0 else t_fill],
                                                         index=[t0_dc_], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc_], name='r'))
                    w_events = w_events.append(pd.Series([int(wd) + int(wd0)], index=[t0_dc_], name='w'))
                    if wd:
                        wd0 = False
                        wd = False
                    if wd0: wd = True
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
        t_1, p_1 = t, p
    duplicated_mask = dc_events.index.duplicated(keep="first")
    print("duplicated = {}".format(sum(duplicated_mask)))
    dc_events = dc_events[~duplicated_mask]
    return pd.DataFrame([dc_events, tmv_events, r_events, t_events, w_events]).T


def direct_change2(s, delta_t, thold=0.02):
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
                p_l = p
                # end time for a downturn event
                t1_dc = t
                # start time for a downward OS
                t0_os = t + delta_t
                dc_events = dc_events.append(pd.Series([p_h, p], index=[t0_dc, t1_dc], name='dc'))
                if dc_events.shape[0] >= 4:
                    tmv_events = tmv_events.append(
                        pd.Series([(p_h - p_l) / (thold * p_l)], index=[t0_dc], name='tmv'))
                    t_events = t_events.append(pd.Series([dc_events.index[-2] - dc_events.index[-4]],
                                                         index=[t0_dc], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc], name='r'))

            elif p_h < p:
                p_h = p
                # start time for a downturn event
                t0_dc = t
                # end time for a upward OS (t-1)
                t1_os = t - delta_t
        else:
            if p >= p_l * (1 + thold):
                upturn = True
                p_h = p
                # end time for a upturn event
                t1_dc = t
                # start time for a upturn OS
                t0_os = t + delta_t
                dc_events = dc_events.append(pd.Series([p_l, p], index=[t0_dc, t], name='dc'))
                if dc_events.shape[0] >= 4:
                    tmv_events = tmv_events.append(
                        pd.Series([(p_h - p_l) / (thold * p_h)], index=[t0_dc], name='tmv'))
                    t_events = t_events.append(pd.Series([dc_events.index[-2] - dc_events.index[-4]],
                                                         index=[t0_dc], name='t'))
                    r_events = r_events.append(pd.Series([(p_h - p_l) / (t_events.iloc[-1] * p_h)],
                                                         index=[t0_dc], name='r'))
            elif p_l > p:
                p_l = p
                # start time for a upturn event
                t0_dc = t
                # end time for a downward OS (t-1)
                t1_os = t - delta_t
    duplicated_mask = dc_events.index.duplicated(keep="first")
    print("duplicated = {}".format(sum(duplicated_mask)))
    dc_events = dc_events[~duplicated_mask]
    return pd.DataFrame([dc_events, tmv_events, r_events, t_events]).T


def cluster_dc(X, n_clusters):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    cluster = KMeans(n_clusters=n_clusters).fit(X)
    labels = cluster.labels_.copy()

    # sorted_t = sorted(np.sum(centroids, axis=1))
    # new_cluster_ix = []

    # for t in sorted_t:
    #     new_cluster_ix.append(np.argmax(np.sum(centroids, axis=1) == t))
    #
    # new_cluster_ix.append(np.argmax(centroids[:,0]))
    # if not np.all(np.diff(new_cluster_ix) >= 0):
    #     for i in range(len(labels)):
    #         labels[i] = new_cluster_ix[labels[i]]

    return labels, cluster, ss


def ema_cluster(dc_df, ema_p, labels, multiplier):
    tmv = dc_df['tmv'].dropna()
    labels_x = pd.DataFrame(labels).astype(int)
    labels_x.index = tmv.index
    labels_x.columns = ['k']
    dc_k = dc_df.join(labels_x)
    dc_k['k'] = dc_k['k'].fillna(method='ffill') * multiplier
    dc_k.dropna(subset=['k'], inplace=True)
    dc_k['k_ema'] = np.rint(ema(dc_k['k'], ema_p))

    return dc_k


def extract_regimes_clusters(dc_df, cfg):
    double_ema, ema2_p, ema_p, map, multiplier, n_clusters = unpack_k_cfg(cfg)
    labels, cluster, ss = get_clusters(n_clusters, dc_df)
    relabel(labels, map)
    dc_k = ema_cluster(dc_df, double_ema, ema_p, labels, multiplier)
    regimes = get_regimes(dc_k['k_ema'])

    return dc_k, regimes, cluster, ss, labels


def unpack_k_cfg(cfg):
    n_clusters, ema_p, = cfg['n_clusters'], cfg['ema_p']
    thold_k, multiplier, map = cfg['thold_k'], cfg['multiplier'], cfg.get('map', None)
    return ema_p, thold_k, multiplier, n_clusters


def new_ix_clusters(cluster):
    centroids = cluster.cluster_centers_
    new_cluster_ix = np.zeros(len(centroids))
    sorted_t = list(range(len(centroids)))
    # remove highest y value
    sorted_t.remove(np.argmax(centroids[:, 1]))
    sorted_t.remove(np.argmax(centroids[:, 0]))

    centroids_left = centroids[sorted_t, :]
    sum_centroids = centroids[:, 1] - centroids[:, 0]
    score = centroids_left[:, 1] - centroids_left[:, 0]
    sorted_score = sorted(score)
    for i in range(len(score)):
        new_cluster_ix[np.where(sum_centroids == sorted_score[i])[0][0]] = i + 1
    # higher y coordinate is the most volatile regime
    new_cluster_ix[np.argmax(centroids[:, 1])] = len(centroids) - 1
    new_cluster_ix[np.argmax(centroids[:, 0])] = 0

    return new_cluster_ix


def get_clusters(n_clusters, dc_df, vars=['t', 'tmv']):
    X = np.array(dc_df.loc[:, vars].dropna())
    labels, cluster, ss = cluster_dc(X, n_clusters)
    return labels, cluster, ss
