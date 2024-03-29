import time
import pandas as pd

from algorithms.hmm.func import resample_dfs
from timeseries.data.market.files.utils import load_files
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.multivariate.architectures.dcnn import dcnn_func
from timeseries.experiments.market.multivariate.architectures.simple import simple_func
from timeseries.experiments.market.preprocess.func import add_features, scale_df
from timeseries.experiments.market.split.func import get_subsets, get_xy, append_subset_cols
from timeseries.experiments.market.utils.data import new_cols_names
from timeseries.experiments.market.utils.dataprep import reshape_xy_prob
from timeseries.experiments.market.utils.filename import find_nth, res_mkt_filename
from timeseries.experiments.market.utils.harness import train_model, test_model, plot_forecast
from timeseries.experiments.market.utils.log import log_forecast_results
from timeseries.experiments.market.utils.regimes import append_state
from timeseries.experiments.market.utils.results import get_results, plot_multiple_results_forecast, confusion_mat
from timeseries.experiments.market.utils.tf_models import save_tf_model, load_tf_model

# split_ES_minute_2012-01_to_2021-06_g4week_r0.25_2021_07_27_11-51
# split_ES_day_2011-12_to_2021-12_g12week_r0.25_2021_07_16_13-16
# split_ES_minute_2018-01_to_2021-06_g4week_r0.25_2021_07_28_10-55
if __name__ == '__main__':
    res_cfg = {'save_results': False, 'save_plots': True, 'plot_title': True, 'plot_forecast': True,
               'plot_hist': False, 'image_folder': 'img', 'results_folder': 'res'}
    # %%
    plot_ = True
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
    training_cfg = {'inst': 'ES', 'y_true_var': 'ESc', 'y_train_var': 'ESc', 'features': [],
                    'use_add_data': False, 'include_ohlc': True, "append_train_to_test": True, 'scale': False,
                    'preprocess': False}

    df_scaled, ss, train_features = scale_df(df, training_cfg)
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
    model_cfg = {'name': 'SIMPLE', 'verbose': 1, 'use_regimes': False,
                 "n_steps_out": 1, "n_steps_in": 1, "n_batch": 24, "n_epochs": 1}

    model_func = simple_func()
    lookback = model_func['lookback'](model_cfg)

    train_x_pp, test_x_pp, features = get_xy(scaled_proba_subsets, training_cfg, lookback, dim_f=1)
    train_data = reshape_xy_prob(model_func, model_cfg, train_x_pp)
    stacked_x, stacked_y, stacked_prob = train_data
    model, train_time, train_loss = train_model(model_cfg, model_func, train_data, summary=False, plot_hist=True)

    # %% TEST
    _, unscaled_test_y, _ = get_xy(unscaled_y_subsets, training_cfg, lookback, dim_f=1)
    test_data = test_x_pp, unscaled_test_y
    t0 = time.time()
    forecast_dfs, metrics, pred_times = test_model(model, model_cfg, training_cfg, model_func,
                                                   test_data, ss, parallel=False)
    print('Pred time: {}s'.format(round(time.time() - t0, 4)))
    all_forecast_df = pd.concat(forecast_dfs, axis=0)

    # %% RESULTS
    results = get_results(metrics, model_cfg, test_x_pp, metric_names=['rmse', 'minmax'], plot_=plot_, print_=True)
    all_forecast_df['rse'] = ((all_forecast_df['data'] - all_forecast_df['forecast']) ** 2) ** .5
    append_state(all_forecast_df, n_states, model_cfg['use_regimes'])
    if plot_:
        file_name = res_mkt_filename(data_mkt_cfg, training_cfg, model_cfg)
        plot_multiple_results_forecast(all_forecast_df, forecast_dfs, model_cfg['use_regimes'], results,
                                       max_subplots=15, n_plots=2, save=res_cfg['save_plots'],
                                       file_path=[res_cfg['image_folder'], file_name])

    # %%
    if plot_:
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
    # %%
