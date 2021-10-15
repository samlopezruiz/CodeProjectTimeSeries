import os

import joblib as joblib
import numpy as np
import pandas as pd

from algorithms.tft2.utils.data import get_col_mapping
from algorithms.tft2.utils.plot import plot_self_attn, plot_historical_attn
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.plotly.plot import plotly_time_series

if __name__ == "__main__":
    #%%
    name = 'snp'
    experiment_name = 'fixed'
    config = ExperimentConfig(name, None)
    result = joblib.load(os.path.join(config.results_folder, experiment_name, 'attention_valid.z'))

    # %%
    attentions, params = result['attentions'], result['params']
    plot_self_attn(attentions, params, taus=[1, 3, 5])
    # plot_historical_attn(attentions, params)
    col_mapping = get_col_mapping(params['column_definition'])
    historical_attn = pd.DataFrame(np.mean(attentions['historical_flags'], axis=0),
                                   columns=col_mapping['historical_inputs'])
    historical_attn.index = np.array(historical_attn.index) - params['num_encoder_steps']
    plotly_time_series(historical_attn, xaxis_title='Position Index (n)')