import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.multivariate.multistep.configs.ensemble import ensemble_mv_configs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import eval_multi_step_forecast
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.models.lorenz.multivariate.multistep.configs.wavenet import wavenet_mv_configs
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'save_results': True, 'verbose': 1, 'plot_title': False, 'plot_hist': False,
              'image_folder': 'forecast_img', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name, input_cfg, model_cfg, functions = ensemble_mv_configs(steps=in_cfg['steps'])
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    #%% WALK FORWARD FORECAST
    data_in = (train_pp, test_pp, train, test, t_train, t_test)
    metrics, forecast = eval_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss,
                                                 label_scale=1.5, size=(1980, 1080 // 2))
