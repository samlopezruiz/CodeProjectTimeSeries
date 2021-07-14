import numpy as np
import pandas as pd
from algorithms.dchange.func import get_regimes, count_regimes, relabel_col
from algorithms.hmm.func import fitHMM
from timeseries.data.market.files.utils import load_market
from timeseries.data.market.regime.plot_utils import hist_vars_state
from timeseries.data.market.utils.time import insert_weekend
from timeseries.plotly.plot import plotly_time_series, plotly_ts_candles, plotly_ts_regime, plotly_histogram_regimes
from timeseries.preprocessing.func import ln_returns
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
    drop_cols = ['volume', 'adl', 'delta']
    df = pd.concat([df_mkt_rsmp, df_fed_rsmp], axis=1)
    df.ffill(axis=0, inplace=True)
    df.dropna(inplace=True)
    df.drop(drop_cols, axis=1, inplace=True)
    df['ES_r'] = ln_returns(df['ESc'])
    df['ES_ar'] = np.abs(df['ES_r'])
    df.dropna(inplace=True)


    # %%
    data_from = '2012-01'
    data_to = '2022-01'
    df_ss = df.loc[data_from:data_to].copy()
    # plotly_ts_candles(df_ss, instrument=mkt_data_cfg['inst'], rows=[i for i in range(df_ss.shape[1] - 4)])

    # %% HMM REGIMES
    save_plots = False
    resample = False
    regime_col = 'state'
    n_states = 4
    vars = ['ES_r', '^VIX', 'macd']  # , 'T10Y2Y', 'T10Y3M'
    X = df_ss.loc[:, vars]
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=n_states)
    df_reg = append_to_df(df_ss, hidden_states, regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)

    df_plot = df_reg.resample('90T').last() if resample else df_reg
    plot_features = ['ESc'] + vars + [regime_col]

    plotly_ts_regime(df_plot, features=plot_features, rows=list(range(len(plot_features))), markers='lines+markers',
                     regime_col=regime_col, markersize=2.5, adjust_height=(True, 0.5), legend=False,
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars)))

    # %% RELABEL
    relabel_col(df_reg, regime_col, map=[0, 2, 1])
    regimes = get_regimes(df_reg[regime_col], print_=True)

    df_plot = df_reg.resample('90T').last() if resample else df_reg
    plotly_ts_regime(df_plot, features=plot_features, rows=list(range(len(plot_features))), markers='lines+markers',
                     regime_col=regime_col, markersize=2.5, adjust_height=(True, 0.5), legend=False,
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars)))

    # %% SAVE
    save_plots = True
    df_plot = df_reg.resample('90T').last() if resample else df_reg
    plotly_ts_regime(df_plot, 'ESc', regime_col, features=vars, adjust_height=(True, 0.6), markersize=4,
                      save=save_plots, file_path=[save_folder, 'hmm'], size=(1980, 1080),
                      title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars)))

