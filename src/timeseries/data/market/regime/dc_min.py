import os

from algorithms.hmm.func import fitHMM
from timeseries.utils.func import interpolate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from timeseries.data.market.utils.time import insert_weekend
from algorithms.dchange.func import direct_change_ts, unpack_k_cfg, get_clusters, relabel, \
    ema_cluster, get_regimes, new_ix_clusters
from timeseries.data.market.load_market import load_market
from timeseries.plotly.plot import plot_dc_clusters, plotly_ts_regime, plotly_time_series
#%%

if __name__ == '__main__':
    #%%
    name = "DIRECTIONAL CHANGE REGIMES"
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'regime_img', 'results_folder': 'results'}
    input_cfg = {'preprocess': True}

    data_cfg = {'inst': "ES", 'suffix': "2012-2020", 'sampling': 'minute',
                'src_folder': "data", 'market': 'cme'}

    # %% LOAD DATA
    es, features_es = load_market(data_cfg)
    insert_weekend(es)

    #%%
    data_from = '2019-12-01 08:30:00'
    data_to = '2021-12-30 15:30:00'
    df = es.loc[data_from:data_to]
    df_h = df.iloc[::90, :]

    # %%
    dc_cfg = {'delta_t': 0.1, 'delta_y': 10}
    dc_df = direct_change_ts(df['ESc'], dc_cfg, df['weekend'])

    final_dc = pd.concat([df_h['ESc'], dc_df['dc']], axis=1)
    # plotly_time_series(dc_df, rows=[i for i in range(dc_df.shape[1])], title=name, markers='lines+markers')
    # plotly_time_series(final_dc, alphas=[.5, 1], title=name, markers='lines+markers')


    #%%
    save_folder = in_cfg['image_folder']
    save_plots = in_cfg['save_results']
    regime_cfg = {'n_clusters': 5, 'ema_p': 150, 'multiplier': 2,  'thold_k': 0.7}

    ema_p, thold_k, multiplier, n_clusters = unpack_k_cfg(regime_cfg)
    labels, cluster, ss = get_clusters(n_clusters, dc_df)
    map = new_ix_clusters(cluster)
    # map = [2, 0, 1, 3]
    relabel(labels, map=map)
    dc_tmv_t = pd.DataFrame(np.array(dc_df.loc[:, ['t', 'tmv']].dropna()), columns=['t', 'tmv'])
    plot_dc_clusters(dc_tmv_t, labels, regime_cfg['n_clusters'], save=save_plots, label_scale=1, markersize=7,
                     title='{}: th={}, k={}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters']),
                     file_path=[save_folder, 'K_{}_th_{}_k_{}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters'])],
                     size=(1980//1.5, 1080//1.5))

    #%%
    dc_k = ema_cluster(dc_df, ema_p=ema_p, multiplier=multiplier, labels=labels)
    dc_k['k_ema'] = ((dc_k['k_ema'] / max(dc_k['k_ema'])) > thold_k).astype(int)

    regimes = get_regimes(dc_k['k_ema'])
    print('regimes changes: {}'.format(len(regimes[1])))
    plotly_ts_regime(dc_k, features=['dc'], markers='lines+markers', regimes=regimes, save=save_plots, regime_col='k_ema',
                     title='{}: k={} t={}'.format(name, regime_cfg['n_clusters'], dc_cfg['delta_y']) + '<br>' + 'REGIME: ' + str(regime_cfg),
                     file_path=[save_folder, '{}_th_{}_k_{}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters'])],
                     size=(1980//1.5, 1080//1.5), markersize=2)

    #%%
    # plotly_ts_regime(df_h, rows=[0, 1], features=['ESc', 'atr'], markers='lines+markers', regimes=regimes, save=save_plots,
    #                  title='{}: k={} t={}'.format(name, regime_cfg['n_clusters'],
    #                                               dc_cfg['delta_y']) + '<br>' + 'REGIME: ' + str(regime_cfg),
    #                  file_path=[save_folder, '{}_th_{}_k_{}'.format(name, dc_cfg['delta_y'], regime_cfg['n_clusters'])],
    #                  size=(1980 // 1.5, 1080 // 1.5), markersize=2)


