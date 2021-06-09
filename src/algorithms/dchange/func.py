import numpy as np
import pandas as pd
import decimal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from timeseries.preprocessing.func import ema


def direct_change(s, cfg):
    thold, delta_t, delta_y = cfg.get('thold', 0.02),  cfg['delta_t'],  cfg['delta_y']
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
                t0_os = t + delta_t
                t0_dc_ = round((t0_dc if t0_dc != dc_events.index[-1] else t0_dc + delta) if len(dc_events) > 0 else t0_dc, round_p)
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
                t0_dc_ = round((t0_dc if t0_dc != dc_events.index[-1] else t0_dc + delta) if len(dc_events) > 0 else t0_dc, round_p)
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
    centroids = cluster.cluster_centers_
    sorted_t = sorted(np.sum(centroids, axis=1))
    new_cluster_ix = []

    for t in sorted_t:
        new_cluster_ix.append(np.argmax(np.sum(centroids, axis=1) == t))

    if not np.all(np.diff(new_cluster_ix) >= 0):
        for i in range(len(labels)):
            labels[i] = new_cluster_ix[labels[i]]

    return labels, cluster, ss


def get_regimes(clusters):
    regimes = [[] for i in range(int(max(clusters) + 1))]
    t_1, k_1 = clusters.index[0], clusters.iloc[0]
    regimes[int(k_1)].append(t_1)

    for t, k in clusters.items():
        if k_1 != k:
            # end of regime
            regimes[int(k_1)].append(t)
            # change of regime
            reg = k
            regimes[int(k)].append(t)

        t_1, k_1 = t, k

    for i in range(len(regimes)):
        regimes[i] = regimes[i][:len(regimes[i]) - len(regimes[i]) % 2]

    return regimes


def extract_regimes(dc_df, labels, ema_p=10, ema2_p=5, double_ema=False, multiplier=1):
    tmv = dc_df['tmv'].dropna()
    labels_x = pd.DataFrame(labels).astype(int)
    labels_x.index = tmv.index
    labels_x.columns = ['k']

    dc_k = dc_df.join(labels_x)
    dc_k['k'] = dc_k['k'].fillna(method='ffill') * multiplier
    dc_k.dropna(subset=['k'], inplace=True)
    dc_k['k_ema'] = np.rint(ema(dc_k['k'], ema_p))
    if double_ema:
        dc_k['k_ema'] = np.rint(ema(dc_k['k_ema'], 5))
    regimes = get_regimes(dc_k['k_ema'])

    return dc_k, regimes


def extract_regimes_clusters(dc_df, cfg):
    n_clusters, ema_p, ema2_p = cfg['n_clusters'], cfg['ema_p'], cfg['ema2_p']
    double_ema, multiplier = cfg['double_ema'], cfg['multiplier']

    X = np.array(dc_df.loc[:, ['t', 'tmv']].dropna())
    labels, cluster, ss = cluster_dc(X, n_clusters)
    dc_k, regimes = extract_regimes(dc_df, labels, ema_p=ema_p, ema2_p=ema2_p,
                                    double_ema=double_ema, multiplier=multiplier)

    return dc_k, regimes, cluster, ss, labels