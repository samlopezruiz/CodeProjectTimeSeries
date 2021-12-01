import os

import joblib
import numpy as np
import pandas as pd
from timeseries.experiments.market.utils.plot import plot_forecast_intervals, group_forecasts
from timeseries.plotly.plot import plotly_time_series

if __name__ == "__main__":
    # %%
    forecast_cfg = {'formatter': 'snp',
                    'experiment_name': '60t_macd',
                    'forecast': 'all_ES_macd_forecasts_1'}

    base_path = os.path.normpath(os.path.join('../outputs/results',
                                              forecast_cfg['formatter'],
                                              forecast_cfg['experiment_name'],
                                              forecast_cfg['forecast']))
    suffix = ''

    results = joblib.load(base_path + suffix + '.z')
    forecasts = results['reconstructed_forecasts'] if 'reconstructed_forecasts' in results else results['forecasts']
    n_output_steps = results['model_params']['total_time_steps'] - results['model_params']['num_encoder_steps']

    identifiers = forecasts['targets']['identifier'].unique()
    target_col = results.get('target', 'ESc')
    results['data'].set_index('datetime', inplace=True)

    data_to_merge = [results['data']]
    for key, df in forecasts.items():
        df.set_index('forecast_time', inplace=True)
        df.columns = ['{} {}'.format(key, col) for col in df.columns]
        data_to_merge.append(df)

    data_w_preds = pd.concat(data_to_merge, axis=1)
    orig_shape = data_w_preds.shape[0]
    data_w_preds.dropna(inplace=True)
    print('Removing {} ({}%) rows that were used as '
          'previous steps in moving window.'.format(orig_shape - data_w_preds.shape[0],
                                                    round(100 * (orig_shape - data_w_preds.shape[0]) /
                                                          data_w_preds.shape[0], 2)))

    #%% Strategy features
    interval_labels = sorted(forecasts.keys())
    if 'p50' in interval_labels:
        interval_labels.remove('p50')
    if 'targets' in interval_labels:
        interval_labels.remove('targets')

    interval_pairs = [(interval_labels[i], interval_labels[len(interval_labels) - 1 - i])
                      for i in range(len(interval_labels) // 2)]

        # use only one pair
    interval_pair = interval_pairs[0]
    lower_bound_labels = ['{} t+{}'.format(interval_pair[0], t + 1) for t in range(n_output_steps)]
    upper_bound_labels = ['{} t+{}'.format(interval_pair[1], t + 1) for t in range(n_output_steps)]
    pred_bound_labels = ['{} t+{}'.format('p50', t + 1) for t in range(n_output_steps)]

    #%%
    df = data_w_preds

    for t in range(n_output_steps):
        df['diff t+{}'.format(t+1)] = df['{} t+{}'.format(interval_pair[1], t+1)] - \
                                              df['{} t+{}'.format(interval_pair[0], t+1)]

    rising_cols = [df[lower_bound_labels[i]] > df[lower_bound_labels[i-1]] for i in range(1, n_output_steps)]
    increment_cols = [(df[lower_bound_labels[i]] - df[lower_bound_labels[i-1]]) / abs(df[lower_bound_labels[i-1]]) for i in range(1, n_output_steps)]


    df['lower_bound_last_pred-first_pred'] = df[lower_bound_labels[-1]] - df[lower_bound_labels[0]]
    df['pred_last_pred-first_pred'] = df[pred_bound_labels[-1]] - df[pred_bound_labels[0]]
    df['upper_bound_last_pred-first_pred'] = df[upper_bound_labels[-1]] - df[upper_bound_labels[0]]
    df['mean_last_pred-first_pred'] = (df['lower_bound_last_pred-first_pred'] + df['pred_last_pred-first_pred'] +
                                       df['upper_bound_last_pred-first_pred']) / 3

    # df['mean_lower_bound_slope'] = pd.concat(increment_cols, axis=1)
    df['lower_bound_rising'] = True
    for rising_col in rising_cols:
        df['lower_bound_rising'] = df['lower_bound_rising'] & rising_col

    print(100 * df['lower_bound_rising'].sum() / len(df['lower_bound_rising']))

    # %%
    plotly_time_series(df.iloc[-2000:, :],
                       features=['ESc',
                                 'ESc_macd_12_24',
                                 'lower_bound_last_pred-first_pred',
                                 'pred_last_pred-first_pred',
                                 'upper_bound_last_pred-first_pred',
                                 'mean_last_pred-first_pred'],
                       rows=[0, 1, 2, 2, 2, 3])

    # %%
    diff_labels = ['diff t+{}'.format(t+1) for t in range(n_output_steps)]
    plotly_time_series(df.iloc[-2000:, :],
                       features=['ESc',
                                 'ESc_macd_12_24',
                                 'pred_last_pred-first_pred'] + diff_labels,
                       rows=[0, 1, 1] + [2 for _ in range(len(diff_labels))])


    #%%
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


    for ix in sorted_ix_cm[:3]:
        id = identifiers[ix]
        plot_forecast_intervals(forecasts_grouped, n_output_steps, id,
                                markersize=3, mode='light',
                                fill_max_opacity=0.1,
                                 title='{}: {}'.format(id,
                                                       np.round(
                                                           results['hit_rates']['grouped_by_id_hit_rate'][ix, Ellipsis],
                                                           4)))
