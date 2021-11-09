from timeseries.experiments.market.expt_settings.configs import ExperimentConfig

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/60t/logs/fit
from timeseries.experiments.market.utils.results import confusion_mat
from timeseries.plotly.plot import plotly_time_series

if __name__ == "__main__":
    # %%
    general_cfg = {'save_forecast': True}

    experiment_cfg = {'formatter': 'snp',
                      'experiment_name': '60t_macd_strategy',
                      'dataset_config': 'ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2015-01_to_2021-06_macd',
                      'vars_definition': 'ES_macd'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()

    mkt_data = formatter.process_data(config.data_config,
                                      scale=False,
                                      indicators_use_time_subset=False,
                                      )

    mkt_data.set_index('datetime', inplace=True)

    # %%
    df_plot = mkt_data.head(1000)
    features = ['ESc', 'ESc_macd_12_24', 'ESc_macd_6_12']
    rows = [0, 1, 1]
    plotly_time_series(df_plot, features=features,
                       markers='lines+markers',
                       rows=rows)

    # %%
    cm, cm_metrics = confusion_mat(y_true=mkt_data['ESc'],
                                   y_pred=mkt_data['ESc_macd_12_24'],
                                   self_change=True,
                                   plot_=True)
