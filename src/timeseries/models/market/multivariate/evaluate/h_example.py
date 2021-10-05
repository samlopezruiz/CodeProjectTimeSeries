import time

import pandas as pd

from algorithms.hmm.func import resample_dfs
from timeseries.data.market.files.utils import load_files
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.models.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.models.market.multivariate.architectures.dcnn import dcnn_func
from timeseries.models.market.multivariate.architectures.simple import simple_func
from timeseries.models.market.preprocess.func import add_features, scale_df
from timeseries.models.market.split.func import get_subsets, get_xy, append_subset_cols
from timeseries.models.market.utils.data import new_cols_names
from timeseries.models.market.utils.dataprep import reshape_xy_prob
from timeseries.models.market.utils.filename import res_mkt_filename
from timeseries.models.market.utils.harness import train_model, test_model, plot_forecast
from timeseries.models.market.utils.log import log_forecast_results
from timeseries.models.market.utils.regimes import append_state
from timeseries.models.market.utils.results import get_results, plot_multiple_results_forecast, confusion_mat
from timeseries.models.market.utils.tf_models import save_tf_model, load_tf_model
# split_ES_minute_2012-01_to_2021-06_g4week_r0.25_2021_07_27_11-51
# split_ES_day_2011-12_to_2021-12_g12week_r0.25_2021_07_16_13-16
# split_ES_minute_2018-01_to_2021-06_g4week_r0.25_2021_07_28_10-55

# OPEN TENSORBOARD IN TERMINAL
# tensorboard - -logdir logs/fit
if __name__ == '__main__':
    res_cfg = {'save_results': False, 'save_plots': False, 'plot_forecast': False, 'plot_title': True,
               'plot_hist': False, 'callbacks': True, 'image_folder': 'img', 'results_folder': 'res',
               'models_folder': "my_models", 'model_version': "0001"}
    data_mkt_cfg = {'filename': "split_ES_minute_2018-01_to_2021-06_g4week_r0.25_2021_07_28_10-55",
                    'src_folder': "res"}
    add_mkt_cfg0 = {'filename': 'subset_NQ_minute_60T_dwn_smpl_2012-01_to_2021-07_2021_08_11_13-08',
                    'src_folder': "res"}
    data_reg_cfg = {'filename': "regime_ESc_r_ESc_macd_T10Y2Y_VIX_2021_07_14_16-29",
                    'src_folder': "res"}

    df, split_cfg, data_cfg_ = load_files(data_mkt_cfg, 'split', end=".z")
    df_add0 = load_files(add_mkt_cfg0, 'additional_data', end=".z")[0]
    df_reg, n_regimes, df_proba, hmm_cfg, data_reg_cfgs_ = load_files(data_reg_cfg, 'regime', end=".z")
    n_states = df_proba.shape[1]

    # %% PREPROCESSING
    training_cfg = {'inst': 'ESc_r', 'y_true_var': 'ESc', 'y_train_var': 'ESc_r', 'features': ['atr', 'ESc_macd'],
                    'use_add_data': True, 'include_ohlc': True, "append_train_to_test": True, 'scale': True,
                    'preprocess': True}

    # append additional features
    add_features(df, macds=['ESc'], returns=get_inst_ohlc_names('ES'))

    # %% ADD DATA
    add_cfg = {'inst': 'NQc_r', 'include_ohlc': True, 'features': ['NQ_atr', 'NQc_macd']}
    if training_cfg['use_add_data']:
        df_add0_resampled = resample_dfs(df, df_add0)
        df_add0_resampled.columns = new_cols_names(df_add0_resampled, 'NQ')
        # append additional features
        add_features(df_add0_resampled, macds=['NQc'], returns=get_inst_ohlc_names('NQ'))

        # get add features
        add_features = (get_inst_ohlc_names(add_cfg['inst']) if add_cfg['include_ohlc'] else []) \
                       + add_cfg['features']
        df_add0_resampled = df_add0_resampled.loc[:, add_features]
        training_cfg['features'] = training_cfg['features'] + add_features
        df_add = pd.concat([df, df_add0_resampled], axis=1)

    df_scaled, ss, train_features = scale_df(df if not training_cfg['use_add_data'] else df_add, training_cfg)
    # scaled data does not contain subset and test columns
    # append subset and test columns to scaled data
    append_subset_cols(df_scaled, df, timediff=True)

    # add hmm probabilites to data
    df_proba_resampled = resample_dfs(df_scaled, df_proba)
    df_scaled_proba_subsets = pd.concat([df_scaled, df_proba_resampled], axis=1)

    # %% SUBSETS
    # compute list of subsets
    scaled_proba_subsets = get_subsets(df_scaled_proba_subsets, n_states=n_states)
    unscaled_y_subsets = get_subsets(df, features=[training_cfg['y_true_var']])

    # %% TRAIN
    # model_cfg = {'name': 'D-CNN', 'verbose': 1, 'use_regimes': True,
    #              "n_steps_out": 1, "n_steps_in": 36, 'n_layers': 4, "n_kernel": 3, 'reg': 'l2',
    #              "n_filters": 64, 'hidden_channels': 9, "n_batch": 24, "n_epochs": 30}
    # model_func = dcnn_func()
    model_cfg = {'name': 'CNN-LSTM', 'verbose': 1, 'use_regimes': True,
                 "n_steps_out": 1, "n_steps_in": 16, "n_seq": 3, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 64, "n_batch": 32, "n_epochs": 10}
    model_func = cnnlstm_func()

    lookback = model_func['lookback'](model_cfg)
    train_x_pp, test_x_pp, features = get_xy(scaled_proba_subsets, training_cfg, lookback, dim_f=1)
    train_data = reshape_xy_prob(model_func, model_cfg, train_x_pp)  # (stacked_x, stacked_y, stacked_prob)
    test_data = reshape_xy_prob(model_func, model_cfg, test_x_pp)

    model, train_time, train_loss = train_model(model_cfg, model_func, train_data, test_data, summary=False,
                                                plot_hist=res_cfg['plot_hist'], callbacks=res_cfg['callbacks'])

    # %% TEST
    _, unscaled_test_y, _ = get_xy(unscaled_y_subsets, training_cfg, lookback, dim_f=1)
    test_forecast_data = test_x_pp, unscaled_test_y
    model_path = save_tf_model(model, [res_cfg['models_folder'], model_cfg['name']], res_cfg['model_version'])
    saved_model = load_tf_model(model_path)
    forecast_dfs, metrics, pred_times = test_model(saved_model, model_cfg, training_cfg, model_func,
                                                   test_forecast_data, ss, parallel=False)
    all_forecast_df = pd.concat(forecast_dfs, axis=0)

    # %% RESULTS
    results = get_results(metrics, model_cfg, test_x_pp, metric_names=['rmse', 'minmax'],
                          plot_=res_cfg['plot_forecast'], print_=True)
    all_forecast_df['rse'] = ((all_forecast_df['data'] - all_forecast_df['forecast']) ** 2) ** .5
    append_state(all_forecast_df, n_states, model_cfg['use_regimes'])
    if res_cfg['plot_forecast']:
        file_name = res_mkt_filename(data_mkt_cfg, training_cfg, model_cfg)
        plot_multiple_results_forecast(all_forecast_df, forecast_dfs, model_cfg['use_regimes'], results,
                                       max_subplots=15, n_plots=2, save=res_cfg['save_plots'],
                                       file_path=[res_cfg['image_folder'], file_name])

    if res_cfg['plot_forecast']:
        file_name = res_mkt_filename(data_mkt_cfg, training_cfg, model_cfg)
        plot_forecast(all_forecast_df, model_cfg, n_states, features=['data', 'forecast'],
                      use_regimes=model_cfg['use_regimes'], markers='markers+lines', adjust_height=(True, 0.8),
                      save=res_cfg['save_plots'], file_path=[res_cfg['image_folder'], file_name])

    # %%
    cm, cm_metrics = confusion_mat(all_forecast_df)
    print('Hit Rate: {} %'.format(round(100 * sum(all_forecast_df['hit_rate']) / all_forecast_df.shape[0], 4)))
    print(cm_metrics)

    log_forecast_results(data_mkt_cfg, data_reg_cfg, training_cfg, model_cfg,
                         results, all_forecast_df, cm_metrics, model_func, train_features)