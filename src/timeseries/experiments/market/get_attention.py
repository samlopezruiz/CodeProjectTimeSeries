import os

import joblib as joblib
import numpy as np
import pandas as pd

from algorithms.tft2.utils.data import get_col_mapping
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.experiments.market.utils.harness import get_model_data_config, get_attention_model
from timeseries.experiments.market.utils.results import process_self_attention, process_historical_vars_attention
from timeseries.experiments.utils.files import save_vars

if __name__ == "__main__":

    general_cfg = {'save_results': True,
                   }

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '5t_ema_q258',
                   'results': 'TFTModel_ES_slow_ema_r_q258_lr001_pred'
                   }

    moo_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(moo_results['experiment_cfg'],
                                                            moo_results['model_cfg'],
                                                            moo_results['fixed_cfg'])

    experiment_cfg = moo_results['experiment_cfg']

    results = get_attention_model(use_gpu=False,
                                  architecture=experiment_cfg['architecture'],
                                  model_folder=model_folder,
                                  data_config=config.data_config,
                                  data_formatter=formatter,
                                  get_attentions=True,
                                  samples=None)

    self_attentions = process_self_attention(results['attentions'],
                                             results['params'],
                                             taus=[1, 3, 5])

    features_attn, mean_hist_attn = process_historical_vars_attention(results['attentions'],
                                                                      results['params'])

    results_processed = {
        'self_attentions': self_attentions,
        'features_attentions': features_attn,
        'mean_features_attentions': mean_hist_attn
    }

    if general_cfg['save_results']:
        print('Saving File')
        save_vars(results_processed,
                  file_path=os.path.join(get_result_folder(results_cfg), 'attention_processed'))
        save_vars(results,
                  file_path=os.path.join(get_result_folder(results_cfg), 'attention_valid'))
    print('Done')
