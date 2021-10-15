import os

from timeseries.experiments.lorenz.multivariate.multistep.dcnnlstm.func import dcnnlstm_get_multi_step_mv_funcs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import eval_multi_step_forecast, run_multi_step_forecast
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.experiments.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results'}

    # MODEL AND TIME SERIES INPUTS
    name = "DCNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 5, 'n_layers': 2, "n_kernel": 2, 'reg': None,
                 "n_filters": 32, 'hidden_channels': 5, "n_batch": 16, "n_epochs": 20, 'n_nodes': 16}
    functions = dcnnlstm_get_multi_step_mv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)

    # %% FORECAST
    # model, forecast_reconst, df = run_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss)

    # %% WALK FORWARD FORECAST
    metrics, forecast = eval_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss)
