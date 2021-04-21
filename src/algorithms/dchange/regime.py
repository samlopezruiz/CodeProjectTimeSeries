from timeseries.data.lorenz.lorenz import multivariate_lorenz
from algorithms.dchange.func import direct_change, extract_regimes_clusters
from timeseries.plotly.plot import plotly_ts_regime, plot_dc_clusters

if __name__ == '__main__':
    # %% INPUTS
    save_folder = 'images'
    save_plots = True
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=1, noise=True, end_time=200,
                                                                  positive_offset=True, trend=True)

    # %% TRANSFORM
    name = "DIRECTIONAL CHANGE REGIMES"
    dc_cfg = {'thold': 0.05, 'delta_t': 0.01}
    regime_cfg = {'n_clusters': 3, 'ema_p': 15, 'multiplier': 2, 'double_ema': False, 'ema2_p': 5}

    dc_df = direct_change(lorenz_df['x'], dc_cfg)
    res = extract_regimes_clusters(dc_df, regime_cfg)
    dc_k, regimes, cluster, ss, labels = res

    plot_dc_clusters(dc_df, labels, regime_cfg['n_clusters'], save=False, label_scale=1, markersize=7,
                     title='{}: th={}, k={}'.format(name, dc_cfg['thold'], regime_cfg['n_clusters']),
                     file_path=[save_folder, 'K_{}_th_{}_k_{}'.format(name, dc_cfg['thold'], regime_cfg['n_clusters'])])

    plotly_ts_regime(lorenz_df, features=['x'], markers='lines', regimes=regimes, save=save_plots,
                     title='{}: k={} t={}'.format(name, regime_cfg['n_clusters'], dc_cfg['thold']) + '<br>' + 'REGIME: ' + str(regime_cfg),
                     file_path=[save_folder, '{}_th_{}_k_{}'.format(name, dc_cfg['thold'], regime_cfg['n_clusters'])])

    # #%%
    # dc_cfgs = []
    # dc_cfgs.append({'thold': 0.07, 'delta_t': 0.01, 'n_clusters': 3, 'ema_p': 5, 'multiplier': 1, 'double_ema': False, 'ema2_p': 5})
    # dc_cfgs.append({'thold': 0.05, 'delta_t': 0.01, 'n_clusters': 3, 'ema_p': 10, 'multiplier': 1, 'double_ema': False, 'ema2_p': 5})
    # dc_cfgs.append({'thold': 0.025, 'delta_t': 0.01, 'n_clusters': 3, 'ema_p': 15, 'multiplier': 1, 'double_ema': False, 'ema2_p': 5})
    #
    # dc_ks = []
    # for dc_cfg in dc_cfgs:
    #     dc_df = direct_change(lorenz_df['x'], dc_cfg)
    #     res = extract_regimes_clusters(dc_df, dc_cfg)
    #     dc_k, regimes, cluster, ss, labels = res
    #     dc_ks.append(dc_k)
    #
    #     plot_dc_clusters(dc_df, labels, regime_cfg['n_clusters'], save=save_plots, label_scale=1, markersize=7,
    #                      title='{}: th={}, k={}'.format(name, dc_cfg['thold'], dc_cfg['n_clusters']),
    #                      file_path=[save_folder,
    #                                 'K_{}_th_{}_k_{}'.format(name, dc_cfg['thold'], dc_cfg['n_clusters'])])
    #
    #     plotly_ts_regime(lorenz_df, features=['x'], markers='lines', regimes=regimes, save=save_plots,
    #                      title='{}: k={}'.format(name, dc_cfg['n_clusters']) + '<br>' + 'REGIME: ' + str(dc_cfg),
    #                      file_path=[save_folder,'{}_th_{}_k_{}'.format(name, dc_cfg['thold'], dc_cfg['n_clusters'])])
    #
    #
    # #%%
    #
    # #%%
    # df0 = dc_ks[2]
    # df1 = dc_ks[1]
    #
    # df21 = sync_df(dc_ks[2], dc_ks[1], features_x=['dc', 'k', 'k_ema'])
    # df_synced = sync_df(df21, dc_ks[0], features_y=['dc', 'k', 'k_ema'], num_inst=2,
    #                     features_x=['dc', 'k', 'k_ema', 'dc_b', 'k_b', 'k_ema_b'])
    #
    # # %%
    # f = 1.5
    # df_all = df_synced.loc[:, ['dc', 'k_ema', 'k_ema_b', 'k_ema_c']]
    # df_all['weigh_k'] = np.rint((df_all['k_ema_b'] + df_all['k_ema_c']) / 2)
    # df_all['weigh_kf'] = np.rint((df_all['k_ema'] * f + df_all['k_ema_b'] * f + df_all['k_ema_c'] * f) / 3)
    # df_all['weigh_k_e'] = np.rint(ema(df_all['weigh_k'], 10))
    # feat = 7
    # plotly_time_series(df_all, rows=list(range(feat)))
    #
    # print(np.sum(np.minimum(np.abs(df_all['k_ema'].diff().dropna()), 1)))
    # print(np.sum(np.minimum(np.abs(df_all['k_ema_b'].diff().dropna()), 1)))
    # print(np.sum(np.minimum(np.abs(df_all['k_ema_c'].diff().dropna()), 1)))
    # print(np.sum(np.minimum(np.abs(df_all['weigh_k'].diff().dropna()), 1)))
    # print(np.sum(np.minimum(np.abs(df_all['weigh_kf'].diff().dropna()), 1)))
    # print(np.sum(np.minimum(np.abs(df_all['weigh_k_e'].diff().dropna()), 1)))
    #
    # #%%
    # regimes = get_regimes(df_all['weigh_k_e'])
    # plotly_ts_regime(lorenz_df, features=['x'], markers='lines', regimes=regimes, save=save_plots,
    #                  title='{}: k={}'.format(name, dc_cfg['n_clusters']) + '<br>' + 'REGIME: ' + str(dc_cfg),
    #                  file_path=[save_folder, '{}_th_{}_k_{}'.format(name, dc_cfg['thold'], dc_cfg['n_clusters'])])




