import numpy as np
import pandas as pd

from algorithms.tft2.utils.data import get_col_mapping
from src.timeseries.plotly.plot import plotly_time_series


def plot_self_attn(attentions,
                   params,
                   taus,
                   label_scale=1,
                   save=False,
                   file_path=None,
                   size=(1980, 1080)):
    self_attentions = []
    # Plot attention for each head
    for i, head_self_attn in enumerate(attentions['decoder_self_attn']):
        self_attn_sample_avg = np.mean(head_self_attn, axis=0)
        n_ts, pred_steps = params['total_time_steps'], params['total_time_steps'] - params['num_encoder_steps']

        self_attn_taus = [pd.Series(self_attn_sample_avg[n_ts - (pred_steps - tau) - 1, :n_ts - (pred_steps - tau)],
                                    name='self_attn t={}'.format(tau)) for tau in taus]
        self_attns = pd.concat(self_attn_taus, axis=1)
        self_attns.index = np.array(self_attns.index) - params['num_encoder_steps']
        self_attentions.append(self_attns)
        plotly_time_series(self_attns,
                           xaxis_title='Position Index (n)',
                           title='Self Attention Head {}'.format(i),
                           label_scale=label_scale,
                           save=save,
                           file_path=file_path+'_head'.format(i),
                           save_png=True,
                           size=size)
    return self_attns


def plot_historical_attn(attentions, params):
    col_mapping = get_col_mapping(params['column_definition'])
    historical_attn = pd.DataFrame(np.mean(attentions['historical_flags'], axis=0),
                                   columns=col_mapping['historical_inputs'])
    historical_attn.index = np.array(historical_attn.index) - params['num_encoder_steps']
    plotly_time_series(historical_attn, xaxis_title='Position Index (n)')
    return historical_attn
