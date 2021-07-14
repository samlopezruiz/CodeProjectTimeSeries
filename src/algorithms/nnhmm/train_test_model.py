import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from timeseries.models.market.utils.harness import train_model, test_model
from timeseries.data.lorenz.lorenz import regime_multivariate_lorenz
from timeseries.models.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.models.market.utils.preprocessing import preprocess

if __name__ == '__main__':
    # %% DATA
    in_cfg = {'use_regimes': True, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_forecast': True,
              'plot_hist': False, 'image_folder': 'images', 'results_folder': 'results'}

    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': False, 'detrend': 'ln_return', 'sigma': 0.5}

    lorenz_df, train, test, t_train, t_test, hidden_states = regime_multivariate_lorenz(input_cfg)
    # plotly_time_series(lorenz_df, features=['x', 'y', 'z'], rows=list(range(3)), markers='lines')
    train_x, train_reg_prob = train
    test_x, test_reg_prob = test

    train_pp, test_pp, train_reg_prob_pp, test_reg_prob_pp, ss = preprocess(input_cfg, train_x, test_x,
                                                                            train_reg_prob, test_reg_prob)

    train_data = (t_train, train_x, train_pp, train_reg_prob, train_reg_prob_pp)
    test_data = (t_test, test_x, test_pp, test_reg_prob, test_reg_prob_pp)

    # %%
    model_cfg = {'name': 'CNNLSTM-REG', "n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}
    n_states = 2
    model_func = cnnlstm_func()

    # %% TRAIN
    model, train_time, train_loss, n_params = train_model(model_cfg, model_func, train_data, summary=False,
                                                          verbose=in_cfg['verbose'], use_regimes=in_cfg['use_regimes'])

    metrics, df, pred_time = test_model(model, input_cfg, model_cfg, model_func, in_cfg, test_data, ss, label_scale=1,
                                        size=(1980, 1080), plot=in_cfg['plot_forecast'], use_regimes=in_cfg['use_regimes'])
