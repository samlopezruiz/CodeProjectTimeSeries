import os

from timeseries.utils.files import new_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.cnn.func import cnn_get_multi_step_mv_funcs
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.data.lorenz.lorenz import lorenz_wrapper

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    model_name = "CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 10, "n_kernel": 5, "n_filters": 128,
                 'n_nodes': 256, "n_batch": 128, "n_epochs": 20}
    func_cfg = cnn_get_multi_step_mv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)

    #%%
    model = func_cfg[2](model_cfg, n_features=3)

    #%%
    new_dir('models')
    model.save('models/' + model_name)
