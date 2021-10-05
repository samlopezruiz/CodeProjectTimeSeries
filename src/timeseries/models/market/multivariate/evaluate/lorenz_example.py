import time

import numpy as np
import pandas as pd
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.market.multivariate.architectures.dcnn import dcnn_func
from timeseries.models.market.preprocess.func import add_features, scale_df
from timeseries.models.market.split.func import get_subsets, get_xy, append_subset_cols
from timeseries.models.market.utils.dataprep import reshape_xy_prob
from timeseries.models.market.utils.harness import train_model, test_model, plot_forecast
from timeseries.models.market.utils.results import subset_results
from timeseries.models.market.utils.tf_models import save_tf_model, load_tf_model
from timeseries.plotly.plot import plotly_time_series


def add_subset_test(df, n_subsets=10, train_ratio=0.7):
    df['subset'] = np.nan
    df['test'] = 0
    ss_size = df.shape[0] // n_subsets
    for ss in range(n_subsets):
        df.iloc[ss * ss_size:(ss + 1) * ss_size, 3] = ss
        df.iloc[ss * ss_size + int(ss_size * train_ratio):(ss + 1) * ss_size, 4] = 1
    df.dropna(inplace=True)


if __name__ == '__main__':
    # %%
    res_cfg = {'save_results': False, 'save_plots': True, 'plot_title': True, 'plot_forecast': True,
               'plot_hist': False, 'image_folder': 'img', 'results_folder': 'res', 'models_folder': "my_models"}
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'trend': True}

    df, _, _, _, _ = lorenz_wrapper(input_cfg)
    n_states = None
    add_subset_test(df, n_subsets=5, train_ratio=0.7)

    # %% PREPROCESSING
    training_cfg = {'inst': None, 'y_true_var': 'x', 'y_train_var': 'x_r', 'features': ['x_r', 'y_r', 'z_r'],
                    'include_ohlc': False, "append_train_to_test": True, 'scale': True}
    # append additional features
    add_features(df, returns=['x', 'y', 'z'])
    df_scaled, ss, _ = scale_df(df, training_cfg)
    # scaled data does not contain subset and test columns
    # append subset and test columns to scaled data
    append_subset_cols(df_scaled, df, timediff=True)

    # %% PREPARE DATA
    # add hmm probabilites to data
    df_scaled_proba_subsets = df_scaled.copy()
    #
    # compute list of subsets
    scaled_proba_subsets = get_subsets(df_scaled_proba_subsets, n_states=n_states)
    unscaled_y_subsets = get_subsets(df, features=[training_cfg['y_true_var']])

    # %% TRAIN
    model_cfg = {'name': 'D-CNN', 'verbose': 1, 'use_regimes': False,
                 "n_steps_out": 3, "n_steps_in": 5, 'n_layers': 2, "n_kernel": 2, 'reg': None,
                 "n_filters": 32, 'hidden_channels': 5, "n_batch": 16, "n_epochs": 20}
    model_func = dcnn_func()
    lookback = model_func['lookback'](model_cfg)

    train_x_pp, test_x_pp, features = get_xy(scaled_proba_subsets, training_cfg, lookback, dim_f=1)
    train_data = reshape_xy_prob(model_func, model_cfg, train_x_pp)
    stacked_x, stacked_y, stacked_prob = train_data
    model, train_time, train_loss = train_model(model_cfg, model_func, train_data)

    #%%
    # model_path = save_tf_model(model, [res_cfg['models_folder'], model_cfg['name']], model_version="0001")
    # saved_model = load_tf_model(model_path)

    # %% TEST
    print('TEST MODEL')
    _, unscaled_test_y, _ = get_xy(unscaled_y_subsets, training_cfg, lookback, dim_f=1)
    test_data = test_x_pp, unscaled_test_y
    t0 = time.time()
    forecast_dfs, metrics, pred_times = test_model(model, model_cfg, training_cfg, model_func, test_data, ss,
                                                   parallel=False)
    print('pred time: {}s'.format(round(time.time() - t0, 4)))

    # %%
    metric_names = ['rmse', 'minmax']
    results = subset_results(metrics, metric_names=metric_names)
    if model_cfg['use_regimes']:
        results['regime'] = [np.mean(np.argmax(prob.to_numpy(), axis=1)) for x, prob in test_x_pp]

    for m in metric_names:
        print('Test {}: {} +-({})'.format(m, round(np.mean(results[m]), 2),
                                          round(np.std(results[m]), 4)))

    rows = list(range(3 if model_cfg['use_regimes'] else 2))
    type_plot = ['bar' for _ in range(3 if model_cfg['use_regimes'] else 2)]
    plotly_time_series(results, rows=rows, xaxis_title='test subset', type_plot=type_plot, plot_ytitles=True)

    # %%
    import seaborn as sns
    import matplotlib.pyplot as plt

    corrMatrix = results.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()

    # %%
    # i = 8
    # forecast_df = forecast_dfs[i]
    # plot_forecast(forecast_df, model_cfg, n_states, metrics[i], use_regimes=model_cfg['use_regimes'],
    #               markers='markers+lines')

    # %%
    all_forecast_df = pd.concat(forecast_dfs, axis=0)
    plot_forecast(all_forecast_df, model_cfg, n_states=0, use_regimes=model_cfg['use_regimes'],
                  markers='markers+lines', save=True, file_path=['img', 'lorenz'])
