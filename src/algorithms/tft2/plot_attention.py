import os

import joblib as joblib

from algorithms.tft.expt_settings.configs import ExperimentConfig
from algorithms.tft.utils.plot import plot_self_attn, plot_historical_attn

if __name__ == "__main__":
    name = 'volatility'
    experiment_name = 'fixed_complete'
    config = ExperimentConfig(name, None)
    result = joblib.load(os.path.join(config.results_folder, experiment_name, 'attention_valid.z'))

    # %%
    attentions, params = result['attentions'], result['params']
    plot_self_attn(attentions, params, taus=[1, 3, 5])
    plot_historical_attn(attentions, params)
