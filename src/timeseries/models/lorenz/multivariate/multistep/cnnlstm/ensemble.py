import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from timeseries.models.utils.config import unpack_in_cfg
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series
from timeseries.models.lorenz.functions.harness import eval_multi_step_forecast, run_multi_step_forecast
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs
# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name = "CNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}
    func_cfg = cnnlstm_get_multi_step_mv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)

    # %% WALK FORWARD FORECAST
    n_ensembles = 5

    forecast_final = None
    for i in range(n_ensembles):
        metrics, forecast = eval_multi_step_forecast(name, input_cfg, model_cfg, func_cfg, in_cfg,
                                                     data_in, ss, plot=False)
        if forecast_final is None:
            forecast_final = forecast
            forecast_final.columns = ['train', 'data', 'forecast0']
        else:
            forecast_final['forecast'+str(i)] = forecast['forecast']

    forecasts = forecast_final.drop(['train', 'data'], axis=1)
    forecast_final['ensemble'] = forecasts.mean(axis=1)
#%%
    minmaxs = []
    for i in range(n_ensembles):
        metrics = forecast_accuracy(forecasts.iloc[:, i].dropna(), forecast_final['data'].dropna())
        minmaxs.append(metrics['minmax'])

    print('individuals minmax = {} +- ({})'.format(round(np.mean(minmaxs), 4), round(np.std(minmaxs), 4)))
    metrics = forecast_accuracy(forecast_final['ensemble'].dropna(), forecast_final['data'].dropna())
    print('ensemble minmax = {}'.format(round(metrics['minmax'], 4)))

#%%
    df = forecast_final.loc[:, ['train', 'data', 'ensemble']]
    size = (1980, 1080)
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    plotly_time_series(df,
                       markers='lines', plot_title=plot_title,
                       file_path=[image_folder, name + "_" + 'ensemble'], save=save_results, size=size,
                       label_scale=1)

    print(metrics)
