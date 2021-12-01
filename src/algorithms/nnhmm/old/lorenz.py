from timeseries.data.lorenz.lorenz import regime_multivariate_lorenz
from timeseries.experiments.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs
from timeseries.experiments.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.experiments.market.utils.harness import eval_forecast
from timeseries.experiments.market.utils.preprocessing import preprocess

if __name__ == '__main__':
    in_cfg = {'steps': 6, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name = "CNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": False, 'preprocess': True,
                 'trend': False, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}


    lorenz_df, train, test, t_train, t_test, hidden_states = regime_multivariate_lorenz(input_cfg)
    # plotly_time_series(lorenz_df, features=['x', 'y', 'z'], rows=list(range(3)), markers='lines')
    train_x, reg_prob_train = train
    test_x, reg_prob_test = test
    train_pp, test_pp, ss = preprocess(input_cfg, train_x, test_x)
    data_in = (train_pp, test_pp, train_x, test_x, t_train, t_test, reg_prob_train, reg_prob_test)

    # %% FORECAST
    # model, forecast_reconst, df = run_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss)

    # %% WITHOUT REGIME
    # func_cfg = cnnlstm_get_multi_step_mv_funcs()
    # metrics, forecast = eval_multi_step_forecast(name, input_cfg, model_cfg, func_cfg, in_cfg, data_in, ss)

    # %% WITHOUT REGIME
    model_func = cnnlstm_func()
    metrics, forecast = eval_forecast(name, input_cfg, model_cfg, model_func, in_cfg, data_in, ss)

    # %% WITHOUT REGIME
    # model_cfg['regime'] = True
    # model_func = cnnlstm_func()
    # metrics, forecast = eval_forecast(name, input_cfg, model_cfg, model_func, in_cfg, data_in, ss)