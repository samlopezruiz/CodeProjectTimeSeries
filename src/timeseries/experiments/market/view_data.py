from timeseries.experiments.market.expt_settings.configs import ExperimentConfig

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/60t/logs/fit
from timeseries.plotly.plot import plotly_time_series

if __name__ == "__main__":
#%%
    general_cfg = {'save_forecast': True}

    experiment_cfg = {'formatter': 'snp',
                  'experiment_name': '60t_ema_test',
                  'dataset_config': 'ES_5t_regime_2015-01_to_2021-06_ema',
                  'vars_definition': 'ES_ema_r',
                  'architecture': 'TFTModel'
                  }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()

    train, valid, test = formatter.split_data(config.data_config,
                                              scale=False,
                                              indicators_use_time_subset=True,
                                              )

    train.set_index('datetime', inplace=True)
    test.set_index('datetime', inplace=True)

    #%%
    df_plot = test.head(1000)
    features = ['ESc', 'ESc_e3', 'ESc_e3_r', 'ESc_r']
    rows = [0, 0, 1, 1]
    plotly_time_series(df_plot, features=features,
                       markers='lines+markers',
                       rows=rows)