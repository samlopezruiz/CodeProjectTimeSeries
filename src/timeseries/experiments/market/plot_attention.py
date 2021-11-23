import os

import joblib as joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms.moo.utils.plot import save_fig
from algorithms.tft2.utils.data import get_col_mapping
from algorithms.tft2.utils.plot import plot_self_attn
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.plotly.plot import plotly_time_series

sns.set_theme('poster')
if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': False,
                   }

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q357',
                   'results': 'attention_valid'
                   }

    result = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    # %%
    attentions, params = result['attentions'], result['params']
    plot_self_attn(attentions,
                   params,
                   taus=[1, 3, 5],
                   label_scale=1.5,
                   save=general_cfg['save_plot'],
                   file_path=os.path.join(get_result_folder(results_cfg),
                                          'img',
                                          'hist_attn_position'),
                   size=(1980 * 2 // 3, 1080 * 2 // 3)
                   )

    col_mapping = get_col_mapping(params['column_definition'])
    historical_attn = pd.DataFrame(np.mean(attentions['historical_flags'], axis=0),
                                   columns=col_mapping['historical_inputs'])
    historical_attn.index = np.array(historical_attn.index) - params['num_encoder_steps']

    mean_hist_attn = historical_attn.mean(axis=0).sort_values(ascending=False).to_frame(name='mean attn')
    sorted_hist_attn = historical_attn.loc[:, mean_hist_attn.index]

    # %%
    n_features_plot = 3
    plotly_time_series(sorted_hist_attn,
                       features=sorted_hist_attn.columns[:n_features_plot],
                       rows=list(range(n_features_plot)),
                       xaxis_title='Position Index (n)',
                       plot_ytitles=True,
                       save=general_cfg['save_plot'],
                       file_path=os.path.join(get_result_folder(results_cfg),
                                              'img',
                                              'hist_attn_position'),
                       label_scale=1.5,
                       save_png=True,
                       size=(1980*2//3, 1080*2//3))

    # %%
    df = mean_hist_attn.copy()
    df['feature'] = df.index
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.barplot(data=df, x='mean attn', ax=ax, y='feature', orient='h')
    plt.tight_layout()
    plt.show()

    if general_cfg['save_plot']:
        save_fig(fig,
                 file_path=os.path.join(get_result_folder(results_cfg),
                                        'img',
                                        'hist_attn'),
                 use_date=False)
