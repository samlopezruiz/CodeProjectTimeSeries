import os

from timeseries.models.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.summarize import summarize_scores
from timeseries.models.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import repeat_evaluate
from timeseries.models.lorenz.functions.preprocessing import preprocess
import time

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'save_results': True, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}
    score_type = 'minmax'
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    n_repeats = 4

    #%%
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    scores = []
    st = time.time()
    for step in [1, 3, 6]:
        name, input_cfg, model_cfg, func_cfg = cnnlstm_mv_configs(steps=step)

        result = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg, func_cfg[0],
                                 func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=0, n_steps_out=in_cfg['steps'])
        metrics, predictions, times, n_params, loss = result
        scores.append(summarize_scores(name, metrics, score_type=score_type))

    print('Grid Search Time: {}'.format(round(time.time() - st, 2)))
