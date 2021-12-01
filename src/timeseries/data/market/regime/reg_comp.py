import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.data.market.files.utils import load_market
from timeseries.utils.dataframes import append_to_df

from algorithms.hmm.func import fitHMM
from timeseries.preprocessing.func import ln_returns
from timeseries.utils.func import interpolate, append_interpolation_ensemble, interpolate_series
import numpy as np
import pandas as pd
from timeseries.data.market.utils.time import insert_weekend
from algorithms.dchange.func import direct_change_ts, unpack_k_cfg, get_clusters, relabel, \
    ema_cluster, get_regimes, new_ix_clusters, count_regimes, relabel_col, cluster_dc
from timeseries.plotly.plot import plot_dc_clusters, plotly_ts_regime, plotly_time_series
from timeseries.preprocessing.func import ema
import matplotlib.pyplot as plt
import seaborn as sns

# %%

if __name__ == '__main__':
    # %%
    name = "DIRECTIONAL CHANGE REGIMES"
    in_cfg = {'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'regime_img', 'results_folder': 'results'}


    data_cfg = {'inst': "ES", 'suffix': "2012_1-2021_6", 'sampling': 'minute',
                'src_folder': "data", 'market': 'cme'}

    save_folder = in_cfg['image_folder']
    save_plots = in_cfg['save_results']

    # %% LOAD MIN and DAY DATA
    es_min, features_es = load_market(data_cfg)
    insert_weekend(es_min)
    data_cfg['sampling'] = 'day'
    data_cfg['suffix'] = '2012_5-2021_6'
    es_day, features_es = load_market(data_cfg)

    # %% SUBSET
    data_from = '2012-01'
    data_to = '2022-01'
    df_up = es_min.loc[data_from:data_to].copy()
    df_down = es_day.loc[data_from:data_to].copy()

    # %% INTERPOLATION ATR
    df_inter = pd.concat([df_up['atr'], df_down['atr']], axis=1)
    df_inter.ffill(axis=0, inplace=True)
    df_inter.columns = ['upsample', 'downsample']
    df_up['atr_day'] = df_inter['downsample']
    df_up.dropna(inplace=True)
    df = df_up.copy()
    # plotly_time_series(df_up, features=['ESc', 'atr', 'atr_day'], rows=[i for i in range(3)])

    # %% Add RETURNS
    df['ES_r'] = ln_returns(df['ESc'])
    df.dropna(inplace=True)

    # %% HMM with Return
    regime_col = 'state'
    n_components = 5
    vars = ['ES_r', 'atr_day']
    X = df.loc[:, vars]
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=n_components)
    df_reg = append_to_df(df, hidden_states, regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)

    df_plot = df_reg.resample('90T').last()
    plotly_ts_regime(df_plot, features=['ESc', 'atr_day', regime_col], rows=[0, 1, 2], markers='lines+markers',
                     regimes=None, regime_col=regime_col, save=save_plots, file_path=[save_folder, 'hmm'],
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_components, str(vars)),
                     size=(1980 // 1.5, 1080 // 1.5), markersize=2.5)

    # %% RELABEL
    regime_col = 'rlbl_state'
    # df_reg[regime_col] = df_reg['state']
    relabel_col(df_reg, regime_col, map=[2, 1, 0])
    # relabel_col(df_reg, 'state', map=[0, 1, 1, 2, 2])
    # regimes = get_regimes(df_reg[regime_col], print_=True)

    df_plot = df_reg.resample('90T').last()
    plotly_ts_regime(df_plot, features=['ESc', 'atr_day', regime_col], rows=[0, 1, 2], markers='lines+markers',
                     regimes=None, regime_col=regime_col, save=save_plots, file_path=[save_folder, 'hmm'],
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_components, str(vars)),
                     size=(1980 // 1.5, 1080 // 1.5), markersize=2.5)

    # %% HMM with DC
    dc_cfg = {'delta_t': 0.1, 'delta_y': 10}
    dc_df = direct_change_ts(df['ESc'], dc_cfg, df['weekend'])
    dc_df['atr_day'] = df['atr_day']
    dc_df['atr_day'].fillna(method='ffill', inplace=True)

    # %%
    regime_col = 'state'
    n_components = 5
    vars = ['tmv', 't', 'atr_day']
    dc_t_tmv = dc_df.loc[:, vars].dropna()
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(dc_t_tmv, 100, n_components=n_components)
    df_reg_t_tmv = append_to_df(dc_t_tmv, hidden_states, regime_col)
    df_reg = append_to_df(dc_df, df_reg_t_tmv['state'], regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)
    # df_plot = df_reg.resample('60T').last()
    plotly_ts_regime(df_reg, features=['dc', 'atr_day', regime_col], rows=[0, 1, 2], markers='lines+markers',
                     regimes=None, regime_col=regime_col, save=save_plots, file_path=[save_folder, 'hmm'],
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_components, str(vars)),
                     size=(1980 // 1.5, 1080 // 1.5), markersize=2.5)
    #%% RELABEL
    relabel_col(df_reg, 'state', map=[0, 1, 2, 2, 3])
    regimes = get_regimes(df_reg[regime_col], print_=True)

    plotly_ts_regime(df_reg, features=['dc', 'atr_day', regime_col], rows=[0, 1, 2], markers='lines+markers',
                     regimes=None, regime_col=regime_col, save=True, file_path=[save_folder, 'hmm'],
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_components, str(vars)),
                     size=(1980 // 1.5, 1080 // 1.5), markersize=2.5)


# #%%
#     new_regime_col = regime_col+'simple'
#     df_reg[new_regime_col] = df_reg['state'].replace(to_replace=1, value=np.nan).fillna(method='ffill')
#     plotly_ts_regime(df_reg, features=['dc', 'atr_day', new_regime_col], rows=[0, 1, 2], markers='lines+markers',
#                      regimes=None, regime_col=new_regime_col, save=save_plots, file_path=[save_folder, 'hmm'],
#                      title='Regime Changes: {}, vars: {}'.format(count_regimes(regimes), str(vars)),
#                      size=(1980 // 1.5, 1080 // 1.5), markersize=2.5)
#
    #%% CLUSTER
    # regime_cfg = {'n_clusters': 5, 'ema_p': 150, 'multiplier': 2,  'thold_k': 0.7}
    # ema_p, thold_k, multiplier, n_clusters = unpack_k_cfg(regime_cfg)
    # vars = ['tmv', 't', 'atr_day']
    #
    # # labels, cluster, ss = get_clusters(n_clusters, dc_df, vars=vars)
    # map = new_ix_clusters(cluster)


    #%% CLUSTER
    regime_col = 'state'
    n_components = 5
    vars = ['tmv', 't', 'atr_day']
    dc_t_tmv = dc_df.loc[:, vars].dropna()
    labels, cluster, ss = cluster_dc(dc_t_tmv.to_numpy(), n_components)
    df_reg_t_tmv = append_to_df(dc_t_tmv, labels, regime_col)
    df_reg = append_to_df(dc_df, df_reg_t_tmv['state'], regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)
    # dc_k = ema_cluster(dc_df, ema_p=ema_p, multiplier=multiplier, labels=labels)

    #%%
    # df_plot = df_reg.resample('60T').last()
    plotly_ts_regime(df_reg, features=['dc', 'atr_day', regime_col], rows=[0, 1, 2], markers='lines+markers',
                     regimes=None, regime_col=regime_col, save=save_plots, file_path=[save_folder, 'hmm'],
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_components, str(vars)),
                     size=(1980 // 1.5, 1080 // 1.5), markersize=2.5)

    # %% RELABEL
    regime_col = 'e_state'
    # df_reg[regime_col] = np.rint(ema(df_reg['state'], 14))
    #
    # # dc_k['k_ema'] = ((dc_k['k_ema'] / max(dc_k['k_ema'])) > thold_k).astype(int)
    # relabel_col(df_reg, regime_col, map=[0,0,1,1,2])
    # regimes = get_regimes(df_reg[regime_col], print_=True)

    plotly_ts_regime(df_reg, features=['dc', 'atr_day', regime_col], rows=[0, 1, 2], markers='lines+markers',
                     regimes=None, regime_col=regime_col, save=True, file_path=[save_folder, 'k_means'],
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_components, str(vars)),
                     size=(1980 // 1.5, 1080 // 1.5), markersize=2.5)

