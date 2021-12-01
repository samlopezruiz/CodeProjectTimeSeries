import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms.moo.utils.plot import plot_runs
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.utils.utils import array_from_lists, write_text_file, latex_table, mean_std_text_df

sns.set_theme('poster')
sns.set_style('dark')

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': True,
                   'save_results': True,
                   'experiment_name': 'q357',
                   'same_experiment': False}

    results_cfg = {'formatter': 'snp'}

    # training_files = [('60t_ema_q357', 'TFTModel_ES_ema_r_q357_lr01_pred'),
    #                   ('60t_ema_q357', 'LSTMModel_ES_ema_r_q357_lr01_pred'),
    #                   ('60t_ema_q357', 'DCNNModel_ES_ema_r_q357_lr01_pred'),
    #                   ]
    # training_files = [('5t_ema_q357', 'TFTModel_ES_slow_ema_r_q357_lr001_pred'),
    #                   ('5t_ema_q357', 'TFTModel_ES_slow_ema_r_q357_lr005_pred'),
    #                   ('5t_ema_q357_vol', 'TFTModel_ES_slow_ema_r_vol_q357_lr001_pred'),
    #                   ]
    # training_files = [('60t_ema_q357', 'TFTModel_ES_ema_r_q357_lr01_pred'),
    #                   ('60t_ema_q357_vol', 'TFTModel_ES_ema_r_vol_q357_lr01_pred'),
    #                   ('60t_ema_q357_svol', 'TFTModel_ES_ema_r_svol_q357_lr01_pred'),
    #                   ]

    training_files = [
        ('60t_ema_q357', 'TFTModel_ES_ema_r_q357_lr01_pred'),
        ('60t_ema_q357_1', 'TFTModel_ES_ema_r_q357_lr01_pred'),
        ('60t_ema_q357_2', 'TFTModel_ES_ema_r_q357_lr01_pred'),
        ('60t_ema_q357_3', 'TFTModel_ES_ema_r_q357_lr01_pred'),
        ('60t_ema_q357_4', 'TFTModel_ES_ema_r_q357_lr01_pred'),
    ]

    label = 'q: 3-5-7'
    experiment_labels = ['run {}'.format(i) for i in range(len(training_files))]

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

    filename = '{}_comp_training'.format(general_cfg['experiment_name'])

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
                                     general_cfg['experiment_name'],
                                     'img',
                                     filename),
              save=general_cfg['save_plot'],
              legend_labels=lbls,
              show_grid=True,
              show_title=False,
              linewidth=5)

    # %%
    val_losses = np.array([nn_result['val_loss'] for nn_result in nn_results]).reshape((-1, 1))
    test_losses = np.delete(np.array([nn_result['test_loss'] for nn_result in nn_results]), 1, axis=1)

    losses = np.hstack([val_losses, test_losses])

    losses_df = pd.DataFrame(losses,
                             columns=['val loss', 'lower q-risk', 'upper q-risk'],
                             index=experiment_labels)
    mean_df = mean_std_text_df(losses_df, label, round=3)

    #%%

    if general_cfg['save_results']:
        write_text_file(os.path.join(results_folder,
                                     save_folder,
                                     'compare',
                                     general_cfg['experiment_name'],
                                     'txt',
                                     '{}'.format(filename)),
                        latex_table('Performance', losses_df.round(3).to_latex()))
        write_text_file(os.path.join(results_folder,
                                     save_folder,
                                     'compare',
                                     general_cfg['experiment_name'],
                                     'txt',
                                     '{}_mean'.format(filename)),
                        latex_table('Performance', mean_df.to_latex()))

    # %%
