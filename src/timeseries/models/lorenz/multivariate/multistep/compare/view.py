import os

from timeseries.models.lorenz.functions.harness import view_multi_step_forecasts
from timeseries.models.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.ensemble import ensemble_mv_configs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.utils.models import get_suffix, get_models_cfgs
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.multivariate.multistep.configs.cnn import cnn_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.models.utils.config import unpack_in_cfg
from timeseries.models.utils.forecast import multi_step_forecasts_df
from timeseries.plotly.plot import plotly_time_series
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.preprocessing import preprocess, reconstruct
from timeseries.models.utils.metrics import forecast_accuracy
# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'save_results': True, 'verbose': 1, 'score_type': 'minmax',
              'plot_title': True, 'plot_hist': False, 'image_folder': 'images', 'results_folder': 'results'}
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    # %% D-CNN & CNN-LSTM
    models = [ensemble_mv_configs,
              dcnn_mv_configs,
              cnnlstm_mv_configs,
              ]

    names, model_cfgs, func_cfgs = get_models_cfgs(models, in_cfg['steps'])
    alphas = [1, 0.9, 1] + [.25] * (len(names) - 1)

    # MODEL AND TIME SERIES INPUTS
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% RUN EVALUATIONS
    data_in = (train_pp, test_pp, train, test, t_train, t_test)
    metrics_res, df = view_multi_step_forecasts(names, input_cfg, model_cfgs,
                                                func_cfgs, in_cfg, data_in, ss, alphas)