import os

from timeseries.experiments.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.configs.wavenet import wavenet_mv_configs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.configs.cnn import cnn_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.ensemble.func import ensemble_get_multi_step_mv_funcs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import eval_multi_step_forecast, run_multi_step_forecast
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name = "ENSEMBLE"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    model_cfg = [convlstm_mv_configs(steps=in_cfg['steps']),
                 dcnn_mv_configs(steps=in_cfg['steps']),
                 wavenet_mv_configs(steps=in_cfg['steps']),
                 cnnlstm_mv_configs(steps=in_cfg['steps']),
                 cnn_mv_configs(steps=in_cfg['steps'])]
    functions = ensemble_get_multi_step_mv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)

    # %% FORECAST
    # model, forecast_reconst, df = run_multi_step_forecast(name, input_cfg, model_cfgs, functions, in_cfg, data_in, ss)

    # %% WALK FORWARD FORECAST
    metrics, forecast = eval_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss)
