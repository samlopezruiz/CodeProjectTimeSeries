import numpy as np
import pandas as pd
from algorithms.dchange.func import get_regimes, count_regimes, relabel_col, direct_change_ts
from algorithms.hmm.func import fitHMM
from timeseries.data.market.files.utils import load_market
from timeseries.data.market.regime.plot_utils import hist_vars_state
from timeseries.data.market.utils.time import insert_weekend
from timeseries.plotly.plot import plotly_time_series, plotly_ts_candles, plotly_ts_regime, plotly_histogram_regimes, \
    plotly_ts_regime2
from timeseries.preprocessing.func import ln_returns
from timeseries.utils.dataframes import append_to_df

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'regime_img', 'results_folder': 'results'}
    save_folder = in_cfg['image_folder']
    save_plots = in_cfg['save_results']
    input_cfg = {'preprocess': True}

    mkt_data_cfg = {'inst': "ES", 'suffix': "2012_1-2021_6", 'sampling': 'minute',
                    'src_folder': "data", 'market': 'cme'}
    df_mkt_up, _ = load_market(mkt_data_cfg)

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
    df_mkt_down = df

    # %%
    data_from = '2012-01'
    data_to = '2022-01'
    df_ss_down = df_mkt_down.loc[data_from:data_to].copy()
    df_ss_up = df_mkt_up.loc[data_from:data_to].copy()


    #%% INTERPOLATION
    add_vars = ['^VIX', 'TEDRATE', 'T10Y2Y', 'T10Y3M', 'atr']
    for var in add_vars:
        df_inter = pd.concat([df_ss_up['ESc'], df_ss_down[var]], axis=1)
        df_inter.ffill(axis=0, inplace=True)
        df_inter.columns = ['upsample', 'downsample']
        df_ss_up[var] = df_inter['downsample']
        df_ss_up.dropna(inplace=True)

    df_ss = df_ss_up.copy()
    df_plot = df_ss.iloc[:4000, :]
    plotly_ts_candles(df_plot, instrument=mkt_data_cfg['inst'], rows=[i for i in range(df_plot.shape[1] - 4)])

    #%% DC
    insert_weekend(df_ss)
    dc_cfg = {'delta_t': 0.1, 'delta_y': 10}
    dc_df = direct_change_ts(df_ss['ESc'], dc_cfg, df_ss['weekend'])
    #%%
    add_vars = ['^VIX', 'TEDRATE', 'T10Y2Y', 'T10Y3M', 'atr']
    dc_df[add_vars] = df_ss[add_vars]
    for var in add_vars:
        dc_df[var].fillna(method='ffill', inplace=True)

    # %% HMM REGIMES
    resample = False
    save_plots = False
    regime_col = 'state'
    n_states = 4
    vars = ['tmv', 't', 'atr']
    dc_t_tmv = dc_df.loc[:, vars].dropna()
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(dc_t_tmv, 100, n_components=n_states)
    df_reg_t_tmv = append_to_df(dc_t_tmv, hidden_states, regime_col)
    df_reg = append_to_df(dc_df, df_reg_t_tmv['state'], regime_col)
    regimes = get_regimes(df_reg[regime_col], print_=True)
    plot_features = ['dc'] + vars + [regime_col]
    df_plot = df_reg.resample('90T').last() if resample else df_reg
    plotly_ts_regime(df_reg, features=plot_features, rows=list(range(len(plot_features))),
                     markers='lines+markers', regime_col=regime_col, markersize=2.5, adjust_height=(True, 0.5),
                     title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars)))

    # %% RELABEL
    # relabel_col(df_reg, regime_col, map=[1,0,2])
    # regimes = get_regimes(df_reg[regime_col], print_=True)
    # df_plot = df_reg.resample('90T').last() if resample else df_reg
    # plotly_ts_regime(df_reg, features=plot_features, rows=list(range(len(plot_features))),
    #                  markers='lines+markers', regime_col=regime_col, markersize=2.5, adjust_height=(True, 0.5),
    #                  title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars)))
    #
    # # %% SAVE
    # save_plots = True
    # df_plot = df_reg.resample('90T').last() if resample else df_reg
    # plotly_ts_regime2(df_plot, 'dc', regime_col, features=vars, adjust_height=(True, 0.6), markersize=2.5,
    #                   save=save_plots, file_path=[save_folder, 'hmm'], size=(1980, 1080),
    #                   title='Regime Changes: {}, k: {}, vars: {}'.format(count_regimes(regimes), n_states, str(vars)))

