import os

import joblib as joblib

from algorithms.tft2.expt_settings.configs import ExperimentConfig
from algorithms.tft2.utils.plot import plot_self_attn, plot_historical_attn

if __name__ == "__main__":
    #%%
    name = 'electricity'
    experiment_name = 'fixed_complete'
    config = ExperimentConfig(name, None)
    result = joblib.load(os.path.join(config.results_folder, experiment_name, 'attention_valid.z'))

    # %%
    attentions, params = result['attentions'], result['params']
    plot_self_attn(attentions, params, taus=[1, 10, 24])
    plot_historical_attn(attentions, params)
