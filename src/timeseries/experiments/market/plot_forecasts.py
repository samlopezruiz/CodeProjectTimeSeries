import os

import joblib
import numpy as np

from timeseries.experiments.market.utils.plot import plot_forecast_intervals, group_forecasts

if __name__ == "__main__":
    # %%
    forecast_cfg = {'formatter': 'snp',
                    'experiment_name': '60t_macd',
                    'forecast': 'TFTModel_ES_fast_macd_forecasts_2'}

    base_path = os.path.join('outputs/results',
                             forecast_cfg['formatter'],
                             forecast_cfg['experiment_name'],
                             forecast_cfg['forecast'])
    suffix = ''

    results = joblib.load(base_path + suffix + '.z')
    forecasts = results['reconstructed_forecasts'] if 'reconstructed_forecasts' in results else results['forecasts']

    identifiers = forecasts['targets']['identifier'].unique()
    target_col = results.get('target', 'ESc')

    # %%
    n_output_steps = results['model_params']['total_time_steps'] - results['model_params']['num_encoder_steps']
    forecasts_grouped = group_forecasts(forecasts, n_output_steps, target_col)

    if results['target']:
        steps = ['{} t+{}'.format(target_col, i + 1) for i in range(n_output_steps)]
    else:
        steps = ['t+{}'.format(i + 1) for i in range(n_output_steps)]

    # %%
    # id = identifiers[5]
    # plot_individual_forecast(forecasts_grouped, n_output_steps, id)
    # plot_forecast_intervals(forecasts_grouped, n_output_steps, id, markersize=3, fill_max_opacity=0.1)

    # %%
    sorted_ix_cm = np.argsort(results['hit_rates']['grouped_by_id_hit_rate'][:, 0, 0] +
                              results['hit_rates']['grouped_by_id_hit_rate'][:, 1, 1])[::-1]


    for ix in sorted_ix_cm[:10]:
        id = identifiers[ix]
        plot_forecast_intervals(forecasts_grouped, n_output_steps, id,
                                markersize=3, mode='light',
                                fill_max_opacity=0.1,
                                 title='{}: {}'.format(id,
                                                       np.round(
                                                           results['hit_rates']['grouped_by_id_hit_rate'][ix, Ellipsis],
                                                           4)))
