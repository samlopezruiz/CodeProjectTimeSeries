import pandas as pd

from algorithms.hmm.func import resample_dfs
from timeseries.data.market.files.utils import load_files
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.models.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.models.market.preprocess.func import add_features, scale_df
from timeseries.models.market.split.func import get_subsets, get_xy, append_subset_cols
from timeseries.models.market.utils.dataprep import reshape_xy_prob
from timeseries.models.market.utils.harness import train_model, test_model, plot_forecast
from timeseries.models.market.utils.regimes import append_state
from timeseries.models.market.utils.results import get_results
from timeseries.plotly.plot import plotly_time_series_bars_hist

if __name__ == '__main__':
    # res_cfg = {'save_results': False, 'plot_title': True, 'plot_forecast': True,
    #            'plot_hist': False, 'image_folder': 'images', 'results_folder': 'results'}
    # %%
    data_mkt_cfg = {'filename': "split_ES_day_2011-12_to_2021-12_g12week_r0.25_2021_07_16_13-16",
                    'src_folder': "res"}
    data_reg_cfg = {'filename': "regime_ESc_r_ESc_macd_T10Y2Y_VIX_2021_07_14_16-29",
                    'src_folder': "res"}

    df, split_cfg, data_cfg_ = load_files(data_mkt_cfg, 'split', end=".z")
    df_reg, n_regimes, df_proba, hmm_cfg, data_reg_cfgs_ = load_files(data_reg_cfg, 'regime', end=".z")
    n_states = df_proba.shape[1]

    # %% PREPROCESSING
    training_cfg = {'inst': 'ESc_r', 'y_true_var': 'ESc', 'y_train_var': 'ESc_r', 'features': ['volume', 'atr'],
                    'include_ohlc': True, "append_train_to_test": True, 'scale': True}
    # append additional features
    add_features(df, macds=['ESc'], returns=get_inst_ohlc_names('ES'))
    df_scaled, ss = scale_df(df, training_cfg)
    # scaled data does not contain subset and test columns
    # append subset and test columns to scaled data
    append_subset_cols(df_scaled, df, timediff=True)

    # %% PREPARE DATA
    # add hmm probabilites to data
    df_proba_resampled = resample_dfs(df_scaled, df_proba)
    df_scaled_proba_subsets = pd.concat([df_scaled, df_proba_resampled], axis=1)

    # compute list of subsets
    scaled_proba_subsets = get_subsets(df_scaled_proba_subsets, n_states=n_states)
    unscaled_y_subsets = get_subsets(df, features=[training_cfg['y_true_var']])

    # %% TRAIN
    model_cfg = {'name': 'CNNLSTM-REG', 'verbose': 1, 'use_regimes': True,
                 "n_steps_out": 3, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 10}
    model_func = cnnlstm_func()
    lookback = model_func['lookback'](model_cfg)

    train_x_pp, test_x_pp, features = get_xy(scaled_proba_subsets, training_cfg, lookback, dim_f=1)
    train_data = reshape_xy_prob(model_func, model_cfg, train_x_pp)
    stacked_x, stacked_y, stacked_prob = train_data
    model, train_time, train_loss = train_model(model_cfg, model_func, train_data, summary=False)

    # %% TEST
    _, unscaled_test_y, _ = get_xy(unscaled_y_subsets, training_cfg, lookback, dim_f=1)
    test_data = test_x_pp, unscaled_test_y
    metrics, forecast_dfs, pred_times = test_model(model, model_cfg, training_cfg, model_func, test_data, ss)
    all_forecast_df = pd.concat(forecast_dfs, axis=0)

    # %% RESULTS
    get_results(metrics, model_cfg, test_x_pp, metric_names=['rmse', 'minmax'], plot_=True, print_=True)

    #%%
    all_forecast_df['rse'] = ((all_forecast_df['data'] - all_forecast_df['forecast']) ** 2) ** .5
    append_state(all_forecast_df, n_states, model_cfg['use_regimes'])
    plotly_time_series_bars_hist(all_forecast_df.loc[:, ['data', 'rse', 'state']], color_col='state')

    # %%
    i = 12
    plot_forecast(forecast_dfs[i], model_cfg, n_states, metrics=metrics[i], use_regimes=model_cfg['use_regimes'],
                  markers='markers+lines', adjust_height=(True, 0.8))

    # %%
    plot_forecast(all_forecast_df, model_cfg, n_states, features=['data', 'forecast'],
                  use_regimes=model_cfg['use_regimes'], markers='markers+lines', adjust_height=(True, 0.8))


