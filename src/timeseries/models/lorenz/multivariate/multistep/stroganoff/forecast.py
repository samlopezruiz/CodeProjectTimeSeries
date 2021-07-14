import os

from algorithms.gpregress.prim import primitives1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.multivariate.multistep.stroganoff.func import stroganoff_get_multi_step_mv_funcs
from timeseries.models.lorenz.functions.harness import run_multi_step_forecast
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.data.lorenz.lorenz import lorenz_wrapper

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    model_name = "STROGANOFF"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 14, "depth": 4, "n_gen": 5, "n_pop": 200,
                 "cxpb": 0.6, "mxpb": 0.07, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5}
    func_cfg = stroganoff_get_multi_step_mv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)

    # %% FORECAST
    model, forecast_reconst, df = run_multi_step_forecast(model_name, input_cfg, model_cfg, func_cfg, in_cfg, data_in, ss)

    # %% WALK FORWARD FORECAST
    # metrics, forecast = eval_multi_step_forecast(model_name, input_cfg, model_cfg, func_cfg, in_cfg, data_in, ss)

    #%%
    from algorithms.stroganoff.standalone import print_tree
    for i in range(model_cfg['n_steps_out']):
        print('____')
        print_tree(model[i].tree)