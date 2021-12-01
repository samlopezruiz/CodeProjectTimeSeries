import pandas as pd

from algorithms.dchange.func import get_regimes, count_regimes, relabel_col
from algorithms.hmm.func import fitHMM, relabel_sort_var, fitHMM_p
from timeseries.data.market.files.utils import load_market
from timeseries.plotly.plot import plotly_ts_regime_hist_vars
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

    fed_data_cfg = {'inst': "FED", 'sampling': 'day', 'suffix': "2012_1-2021_6", 'market': 'fed',
                    'src_folder': "data"}
    df_fed, _ = load_market(fed_data_cfg)

    # %%
    df_mkt_rsmp = df_mkt.resample('D').last()
    df_fed_rsmp = df_fed.resample('D').last()

    # %%
    drop_cols = ['volume', 'atr', 'adl', 'delta']
    df = pd.concat([df_mkt_rsmp, df_fed_rsmp], axis=1)
    df.ffill(axis=0, inplace=True)
    df.dropna(inplace=True)
    df.drop(drop_cols, axis=1, inplace=True)
    df['atr'] = df_mkt_rsmp['atr']
    df['volume'] = df_mkt_rsmp['volume']
    df['ES_r'] = ln_returns(df['ESc'])
    df['macd'] = macd(df['ESc'])
    df.dropna(inplace=True)

    # %%
    data_from = '2012-01'
    data_to = '2022-01'
    df_ss = df.loc[data_from:data_to].copy()

    # %% HMM REGIMES
    save_plots = False
    resample = False
    regime_col = 'state'
    n_states = 4
    vars = ['ES_r', 'macd', '^VIX']  # , 'T10Y2Y', 'T10Y3M' '^VIX', 'macd', 'DGS10',
    label_cfg = ('macd', 'min')

    X = df_ss.loc[:, vars]
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 500, n_components=n_states)
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
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM_p(X, 500, n_components=n_states)
    df_reg = append_to_df(df_ss, hidden_states, regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)
    # map = relabel_sort_var(label_cfg, vars, mus)
    # relabel_col(df_reg, regime_col, map=map)

#%%
    df_plot = df_reg.resample('90T').last() if resample else df_reg
    plot_features = ['ESc'] + vars + [regime_col]

    title = 'Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars))
    name = 'hmm_{}'.format('_' + '_'.join(vars))
    plotly_ts_regime_hist_vars(df_plot, 'ESc', regime_col, features=vars, adjust_height=(True, 0.6), markersize=4,
                               save=save_plots, file_path=[save_folder, name], size=(1980, 1080), title=title)
