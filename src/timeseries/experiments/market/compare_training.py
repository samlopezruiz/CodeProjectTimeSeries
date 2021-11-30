import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms.moo.utils.plot import plot_runs
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.utils.utils import array_from_lists, write_text_file, latex_table

sns.set_theme('poster')
sns.set_style('dark')

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': False,
                   'save_results': False,
                   'test_name': 'architectures',
                   'same_experiment': True}

    results_cfg = {'formatter': 'snp'}

    # training_files = [('60t_ema_q159', 'TFTModel_ES_ema_r_q159_lr01_pred'),
    #                   ('60t_ema_q159', 'LSTMModel_ES_ema_r_q159_lr01_pred'),
    #                   ('60t_ema_q159', 'DCNNModel_ES_ema_r_q159_lr01_pred'),
    #                   ]
    # training_files = [('5t_ema_q258', 'TFTModel_ES_slow_ema_r_q258_lr001_pred'),
    #                   ('5t_ema_q258', 'TFTModel_ES_slow_ema_r_q258_lr005_pred'),
    #                   ('5t_ema_q258_vol', 'TFTModel_ES_slow_ema_r_vol_q258_lr001_pred'),
    #                   ]
    # training_files = [('60t_ema_q258', 'TFTModel_ES_ema_r_q258_lr01_pred'),
    #                   ('60t_ema_q258_vol', 'TFTModel_ES_ema_r_vol_q258_lr01_pred'),
    #                   ('60t_ema_q258_svol', 'TFTModel_ES_ema_r_svol_q258_lr01_pred'),
    #                   ]
    training_files = [('60t_ema_q258', 'TFTModel_ES_ema_r_q258_lr01_pred'),
                      ('60t_ema_q258_svol', 'TFTModel_ES_ema_r_svol_q258_lr01_pred'),
                      ('60t_ema_q258_vol', 'TFTModel_ES_ema_r_vol_q258_lr01_pred'),
                      ]


    experiment_labels = ['60t_ema_q258', '60t_ema_q258_svol', '60t_ema_q258_vol']

    # training_files = [('60t_ema_q258', 'TFTModel_ES_ema_r_q258_lr01_pred_old'),
    #                   ('60t_ema_q258', 'TFTModel_ES_ema_r_q258_lr01_pred'),
    #                   ]

    # training_files = [('60t_ema_q159_test', 'TFTModel_ES_ema_r_q159_lr01_pred'),
    #                   ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_lr01_pred'),
    #                   # ('60t_ema_q159', 'DCNNModel_ES_ema_r_q159_lr01_pred'),
    #                   ]

    save_folder = training_files[0][0] if general_cfg['same_experiment'] else ''

    results_folder = get_result_folder(results_cfg)
    nn_results = [joblib.load(os.path.join(results_folder, *file) + '.z') for file in training_files]

    # experiment_labels = ['{}'.format(nn_result['experiment_cfg']['architecture'].replace('Model', ''))
    #                      for nn_result in nn_results]

    # experiment_labels = ['{} lr:{}'.format(nn_result['experiment_cfg']['architecture'].replace('Model', ''),
    #                                        nn_result['model_params']['learning_rate'])]
    # experiment_labels = ['batch:{}'.format(nn_result['model_params']['minibatch_size'])
    #                      for nn_result in nn_results]

    # experiment_labels = ['{}'.format(nn_result['experiment_cfg']['vars_definition'])
    #                      for nn_result in nn_results]



    filename = '{}_comp_training'.format(general_cfg['test_name'])

    # %%
    train_hist = array_from_lists([nn_result['fit_history']['loss'] for nn_result in nn_results])
    val_hist = array_from_lists([nn_result['fit_history']['val_loss'] for nn_result in nn_results])

    lbls = ['{} {}'.format(exp_lbl, l) for l in ['train', 'val'] for exp_lbl in experiment_labels]
    plot_runs([train_hist, val_hist],
              mean_run=None,
              x_label='Epoch',
              y_label='Loss',
              title='Loss History',
              size=(15, 12),
              file_path=os.path.join(results_folder,
                                     save_folder,
                                     'compare',
                                     'img',
                                     filename),
              save=general_cfg['save_plot'],
              legend_labels=lbls,
              show_grid=True,
              show_title=True,
              linewidth=5)

    #%%
    val_losses = np.array([nn_result['val_loss'] for nn_result in nn_results]).reshape((-1, 1))
    test_losses = np.delete(np.array([nn_result['test_loss'] for nn_result in nn_results]), 1, axis=1)

    losses = np.hstack([val_losses, test_losses])

    losses_df = pd.DataFrame(losses,
                             columns=['val loss', 'lower q-risk', 'upper q-risk'],
                             index=experiment_labels)

    if general_cfg['save_results']:
        write_text_file(os.path.join(results_folder,
                                     save_folder,
                                     'compare',
                                     'txt',
                                     '{}'.format(filename)),
                        latex_table('Performance', losses_df.round(3).to_latex()))

    #%%

