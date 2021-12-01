import numpy as np
import pandas as pd
from algorithms.dchange.func import get_regimes, count_regimes, relabel_col
from algorithms.hmm.func import fitHMM, relabel_sort_var
from timeseries.data.market.files.utils import load_market
from timeseries.data.market.regime.plot_utils import hist_vars_state
from timeseries.data.market.utils.time import insert_weekend
from timeseries.plotly.plot import plotly_time_series, plotly_ts_candles, plotly_ts_regime, plotly_histogram_regimes, \
    plotly_ts_regime_hist_vars
from timeseries.preprocessing.func import ln_returns, macd
from timeseries.utils.dataframes import append_to_df

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'regime_img', 'results_folder': 'results'}
    save_folder = in_cfg['image_folder']
    save_plots = in_cfg['save_results']
    input_cfg = {'preprocess': True}

    mkt_data_cfg = {'inst': "ES", 'suffix': "2012_5-2021_6", 'sampling': 'day',
                    'src_folder': "data", 'market': 'cme'}
    df_mkt, _ = load_market(mkt_data_cfg)

    mkt_data_cfg2 = {'inst': "NQ", 'suffix': "2012_5-2021_6", 'sampling': 'day',
                     'src_folder': "data", 'market': 'cme'}
    df_mkt2, _ = load_market(mkt_data_cfg2)

    fed_data_cfg = {'inst': "FED", 'sampling': 'day', 'suffix': "2012_1-2021_6", 'market': 'fed',
                    'src_folder': "data"}
    df_fed, _ = load_market(fed_data_cfg)

    # %%
    df_mkt_rsmp = df_mkt.resample('D').last()
    df_mkt_rsmp2 = df_mkt2.resample('D').last()
    df_fed_rsmp = df_fed.resample('D').last()

    # %%
    drop_cols = ['volume', 'atr', 'adl', 'delta']
    df = pd.concat([df_mkt_rsmp, df_fed_rsmp, df_mkt_rsmp2], axis=1)
    df.ffill(axis=0, inplace=True)
    df.dropna(inplace=True)
    df.drop(drop_cols, axis=1, inplace=True)
    df['atr'] = df_mkt_rsmp['atr']
    df['volume'] = df_mkt_rsmp['volume']
    df['ESh_r'] = ln_returns(df['ESh'])
    df['ESl_r'] = ln_returns(df['ESl'])
    df['ES_r'] = ln_returns(df['ESc'])
    df['NQ_r'] = ln_returns(df['NQc'])
    df['ES_ar'] = np.abs(df['ES_r'])
    df['macd'] = macd(df['ESc'])
    df.dropna(inplace=True)

    # %%
    data_from = '2012-01'
    data_to = '2022-01'
    df_ss = df.loc[data_from:data_to].copy()
    # features = ['NQc', 'volume', 'atr', 'ES_r', '^VIX', 'DGS10']
    # plotly_ts_candles(df_ss, features=features, instrument=mkt_data_cfg['inst'], rows=[i for i in range(len(features))])

    # %% HMM REGIMES
    save_plots = False
    resample = False
    regime_col = 'state'
    n_states = 4
    vars = ['ES_r', 'macd', 'T10Y2Y', '^VIX']  # , 'T10Y2Y', 'T10Y3M' '^VIX', 'macd', 'DGS10',
    label_cfg = ('macd', 'min')

    s = int(df_ss.shape[0] * .9)
    df_train = df_ss.iloc[:s, :].copy()
    df_test = df_ss.iloc[s:, :].copy()

    #%%
    X = df_ss.loc[:, vars]
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=n_states)
    df_reg = append_to_df(df_ss, hidden_states, regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)
    map = relabel_sort_var(label_cfg, vars, mus)
    relabel_col(df_reg, regime_col, map=map)

    df_plot = df_reg.resample('90T').last() if resample else df_reg
    plot_features = ['ESc'] + vars + [regime_col]

    title = 'Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars))
    name = 'hmm_{}'.format('_' + '_'.join(vars))
    plotly_ts_regime_hist_vars(df_plot, 'ESc', regime_col, features=vars, adjust_height=(True, 0.6), markersize=4,
                               save=save_plots, file_path=[save_folder, name], size=(1980, 1080), title=title)

    #%%
    # X_train = df_train.loc[:, vars]
    # hidden_states_train, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X_train, 100, n_components=n_states)
    # df_reg = append_to_df(df_train, hidden_states_train, regime_col)
    # regimes = get_regimes(df_reg[regime_col], print_=True)
    # map = relabel_sort_var(label_cfg, vars, mus)
    # relabel_col(df_reg, regime_col, map=map)
    #
    # df_plot = df_reg.resample('90T').last() if resample else df_reg
    # plot_features = ['ESc'] + vars + [regime_col]
    #
    # title = 'Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars))
    # name = 'hmm_{}'.format('_' + '_'.join(vars))
    # plotly_ts_regime_hist_vars(df_plot, 'ESc', regime_col, features=vars, adjust_height=(True, 0.6), markersize=4,
    #                   save=save_plots, file_path=[save_folder, name], size=(1980, 1080), title=title)


    #%%
    X_train = df_train.loc[:, vars]
    shape = np.array(X_train).shape
    obs = np.argmax(shape)
    f = np.argmin(shape)
    Q = np.reshape(X_train, [shape[obs], shape[f]])
    hidden_states_train = model.predict(Q)

    X_test = df_test.loc[:, vars]
    shape = np.array(X_test).shape
    obs = np.argmax(shape)
    f = np.argmin(shape)
    Q = np.reshape(X_test, [shape[obs], shape[f]])
    hidden_states_test = model.predict(Q)
    # hidden_states2 = model.decode(Q)

    #%%
    hidden_states = np.concatenate((hidden_states_train, hidden_states_test))
    df_reg = append_to_df(df_ss, hidden_states, regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)
    map = relabel_sort_var(label_cfg, vars, mus)
    relabel_col(df_reg, regime_col, map=map)

    df_plot = df_reg.resample('90T').last() if resample else df_reg
    plot_features = ['ESc'] + vars + [regime_col]

    title = 'Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars))
    name = 'hmm_{}'.format('_' + '_'.join(vars))
    plotly_ts_regime_hist_vars(df_plot, 'ESc', regime_col, features=vars, adjust_height=(True, 0.6), markersize=4,
                               save=save_plots, file_path=[save_folder, name], size=(1980, 1080), title=title)
