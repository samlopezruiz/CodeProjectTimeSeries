import os

from timeseries.data.market.files.utils import load_market

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.hmm.func import fitHMM
from timeseries.preprocessing.func import ln_returns
from timeseries.utils.func import interpolate, append_interpolation_ensemble, interpolate_series


import numpy as np
import pandas as pd
from timeseries.data.market.utils.time import insert_weekend
from algorithms.dchange.func import direct_change_ts, unpack_k_cfg, get_clusters, relabel, \
    ema_cluster, get_regimes, new_ix_clusters
from timeseries.plotly.plot import plot_dc_clusters, plotly_ts_regime, plotly_time_series
from timeseries.preprocessing.func import ema
import matplotlib.pyplot as plt
import seaborn as sns
#%%

def smooth_probs(df, e_period=5):
    df['e_proba0'] = ema(df['proba0'], e_period)
    df['e_proba1'] = ema(df['proba1'], e_period)
    e_probas = df.loc[:, ['e_proba0', 'e_proba1']].to_numpy()
    e_hidden = np.zeros(e_probas.shape[0]).astype(int)
    for i, row in enumerate(e_probas):
        e_hidden[i] = np.argmax(row)
    df['e_state'] = e_hidden

if __name__ == '__main__':
    #%%
    name = "DIRECTIONAL CHANGE REGIMES"
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'regime_img', 'results_folder': 'results'}
    input_cfg = {'preprocess': True}

    data_cfg = {'inst': "ES", 'suffix': "2012-2020", 'sampling': 'minute',
                'src_folder': "data", 'market': 'cme'}

    # %% LOAD MIN and DAY DATA
    es_min, features_es = load_market(data_cfg)
    insert_weekend(es_min)
    data_cfg['sampling'] = 'day'
    es_day, features_es = load_market(data_cfg)

    #%% SUBSET
    data_from = '2016-12-01 08:30:00'
    data_to = '2021-01-28 15:30:00'
    df_up = es_min.loc[data_from:data_to].copy()
    df_down = es_day.loc[data_from:data_to].copy()

    # %% INTERPOLATION ATR
    df_inter = pd.concat([df_up['atr'], df_down['atr']], axis=1)
    df_inter.ffill(axis=0, inplace=True)
    df_inter.columns = ['upsample', 'downsample']
    df_up['atr_day'] = df_inter['downsample']
    df_up.dropna(inplace=True)
    # plotly_time_series(df_up, features=['ESc', 'atr', 'atr_day'], rows=[i for i in range(3)])
    df = df_up.copy()
    df_h = df_up.iloc[::90, :]
    #%% HMM
    df['ES_r'] = ln_returns(df['ESc'])
    df.dropna(inplace=True)

    #%%
    vars = ['ES_r', 'atr', 'atr_day']
    X = df.loc[:, vars]
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=2)

    # plotly_time_series(df, features=['ESc', 'proba0', 'proba1'], rows=[i for i in range(3)])

    df['state'] = hidden_states
    regimes = get_regimes(df['state'])
    print('regimes changes: {}'.format(len(regimes[1])))

    #%%
    e_period = 5
    df[['proba0', 'proba1']] = hidden_proba
    smooth_probs(df, e_period)
    regimes = get_regimes(df['e_state'])
    print('regimes changes: {}'.format(len(regimes[1])))
#%%
    df['ema_state'] = (ema(df['state'].astype(float), e_period) > 0.5).astype(int)
    regimes = get_regimes(df['ema_state'])
    print('regimes changes: {}'.format(len(regimes[1])))

    #%%
    save_folder = in_cfg['image_folder']
    save_plots = in_cfg['save_results']
    # plotly_ts_regime(df, features=['ESc', 'atr', 'atr_day'], rows=[0, 1, 2], markers='lines+markers', regimes=None, save=save_plots,
    #                  regime_col='state', file_path=[save_folder, 'hmm'], size=(1980 // 1.5, 1080 // 1.5), markersize=2)
    #%%


    # plotly_ts_regime(df_h, features=['ESc', 'atr_day'], rows=[0, 1], markers='lines+markers', regimes=None,
    #                  save=save_plots, regime_col=None, file_path=[save_folder, 'hmm'], size=(1980 // 1.5, 1080 // 1.5), markersize=2)

    #%% DIRECTIONAL CHANGE
    dc_cfg = {'delta_t': 0.1, 'delta_y': 10}
    dc_df = direct_change_ts(df['ESc'], dc_cfg, df['weekend'])
    dc_df['atr_day'] = df['atr_day']
    dc_df['atr_day'].fillna(method='ffill', inplace=True)
    final_dc = pd.concat([df_h['ESc'], dc_df['dc']], axis=1)
    # plotly_time_series(dc_df, rows=[i for i in range(dc_df.shape[1])], title=name, markers='lines+markers')
    # plotly_time_series(final_dc, alphas=[.5, 1], title=name, markers='lines+markers')

    #%%
    n_components = 3
    vars = ['tmv', 't', 'atr_day']
    df_x = dc_df.loc[:, vars].dropna()
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(df_x, 100, n_components=n_components)

    # plotly_time_series(df, features=['ESc', 'proba0', 'proba1'], rows=[i for i in range(3)])

    df_x['state'] = hidden_states

    # %%
    sns.scatterplot(data=df_x, y='tmv', x='t', hue='state')
    plt.show()

    # %%
    labels = df_x['state'].to_numpy()
    map = [2, 0, 1]
    # map = [2, 1, 0]
    relabel(labels, map=map)
    df_x['state'] = labels

    #%%
    # df_x[['proba0', 'proba1']] = hidden_proba
    # smooth_probs(df_x, e_period=10)
    # regimes = get_regimes(df_x['e_state'])
    # print('regimes changes: {}'.format(len(regimes[1])))

    df_reg = pd.concat([dc_df, df_x.loc[:, ['state']]], axis=1)
    df_reg['state'] = df_reg['state'].fillna(method='ffill')
    mask = ~df_reg['state'].isna()
    df_reg = df_reg.iloc[mask.values, :].copy()
    df_reg['e_state'] = (ema(df_reg['state'].astype(float), e_period) > 0.9).astype(int)

    #%%
    regime_col = 'state'
    reg_chgs = 0
    regimes = get_regimes(df_reg[regime_col])
    for reg in regimes:
        reg_chgs += len(reg)
    print('regimes changes: {}'.format(reg_chgs))

    plotly_ts_regime(df_reg, features=['dc', 'atr_day', regime_col], rows=[0, 1, 2], markers='lines+markers', regimes=None,  regime_col=regime_col,
                     title='Regime Changes: {}'.format(reg_chgs), save=save_plots, file_path=[save_folder, 'hmm'], size=(1980//1.5, 1080//1.5), markersize=2)

    #%%

    #%%
    # save_folder = in_cfg['image_folder']
    # save_plots = in_cfg['save_results']
    # regime_cfg = {'n_clusters': 5, 'ema_p': 150, 'multiplier': 2,  'thold_k': 0.7}
    #
    # ema_p, thold_k, multiplier, n_clusters = unpack_k_cfg(regime_cfg)
    # labels, cluster, ss = get_clusters(n_clusters, dc_df)
    # map = new_ix_clusters(cluster)
    # # map = [2, 0, 1, 3]
    # relabel(labels, map=map)
    # dc_tmv_t = pd.DataFrame(np.array(dc_df.loc[:, ['t', 'tmv']].dropna()), columns=['t', 'tmv'])
    # plot_dc_clusters(dc_tmv_t, labels, regime_cfg['n_clusters'], save=save_plots, label_scale=1, markersize=7,
    #                  title='{}: th={}, k={}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters']),
    #                  file_path=[save_folder, 'K_{}_th_{}_k_{}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters'])],
    #                  size=(1980//1.5, 1080//1.5))
    #
    # #%%
    # dc_k = ema_cluster(dc_df, ema_p=ema_p, multiplier=multiplier, labels=labels)
    # dc_k['k_ema'] = ((dc_k['k_ema'] / max(dc_k['k_ema'])) > thold_k).astype(int)
    #
    # regimes = get_regimes(dc_k['k_ema'])
    # print('regimes changes: {}'.format(len(regimes[1])))
    # plotly_ts_regime(dc_k, features=['dc'], markers='lines+markers', regimes=regimes, save=save_plots, regime_col='k_ema',
    #                  title='{}: k={} t={}'.format(name, regime_cfg['n_clusters'], dc_cfg['delta_y']) + '<br>' + 'REGIME: ' + str(regime_cfg),
    #                  file_path=[save_folder, '{}_th_{}_k_{}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters'])],
    #                  size=(1980//1.5, 1080//1.5), markersize=2)

    #%%
    # plotly_ts_regime(df_h, rows=[0, 1], features=['ESc', 'atr'], markers='lines+markers', regimes=regimes, save=save_plots,
    #                  title='{}: k={} t={}'.format(name, regime_cfg['n_clusters'],
    #                                               dc_cfg['delta_y']) + '<br>' + 'REGIME: ' + str(regime_cfg),
    #                  file_path=[save_folder, '{}_th_{}_k_{}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters'])],
    #                  size=(1980 // 1.5, 1080 // 1.5), markersize=2)


