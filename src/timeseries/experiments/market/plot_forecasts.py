import os

import joblib
import numpy as np

from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.plot import plot_forecast_intervals, group_forecasts

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': False,
                   'plot_title': False}

    forecast_cfg = {'formatter': 'snp',
                    'experiment_name': '60t_ema_q258',
                    'forecast': 'TFTModel_ES_ema_r_q258_lr01_pred',
                    'subfolders': []
                    }
                    # 'subfolders': ['moo', 'selec_sols']}

    additional_vars = ['ESc']

    base_path = os.path.join('outputs/results',
                             forecast_cfg['formatter'],
                             forecast_cfg['experiment_name'],
                             *forecast_cfg['subfolders'],
                             forecast_cfg['forecast'])
    suffix = ''

    results = joblib.load(base_path + suffix + '.z')

    if len(additional_vars) > 0:
        config = ExperimentConfig(results['experiment_cfg']['formatter'], results['experiment_cfg'])
        formatter = config.make_data_formatter()
        mkt_data, add_data, reg_data = formatter.load_data(config.data_config)

    forecasts = results['reconstructed_forecasts'] if 'reconstructed_forecasts' in results else results['forecasts']

    identifiers = forecasts['targets']['identifier'].unique()
    target_col = results.get('target', 'ESc')

    n_output_steps = results['model_params']['total_time_steps'] - results['model_params']['num_encoder_steps']
    forecasts_grouped = group_forecasts(forecasts, n_output_steps, target_col)

    if results['target']:
        steps = ['{} t+{}'.format(target_col, i + 1) for i in range(n_output_steps)]
    else:
        steps = ['t+{}'.format(i + 1) for i in range(n_output_steps)]

    # sorted_ix_cm = np.argsort(results['hit_rates']['grouped_by_id_hit_rate'][:, 0, 0] +
    #                           results['hit_rates']['grouped_by_id_hit_rate'][:, 1, 1])[::-1]

    img_path = os.path.join('outputs/results',
                             forecast_cfg['formatter'],
                             forecast_cfg['experiment_name'],
                             'img',
                            *forecast_cfg['subfolders'])

    filename = forecast_cfg['forecast']

    #%%
    plot_segments = []
    plot_segments.append({'sorted_ix_cm': 1,
                          'y_range': [2000, 2050],
                          'x_range': ['2015-04-19T20:00', '2015-04-24T15:00']})
    plot_segments.append({'sorted_ix_cm': 20,
                          'y_range': [2850, 2970],
                          'x_range': ['2019-09-01T21:00', '2019-09-06T17:00']})
    plot_segments.append({'sorted_ix_cm': 27,
                          'y_range': [4000, 4250],
                          'x_range': ['2021-05-09T21:00', '2021-05-14T17:00']})


    for plot_segment in plot_segments:
        id = identifiers[plot_segment['sorted_ix_cm']]
        title = 'Filename: {} <br>Model: {}, Vars Definition: {},' \
                '<br>Dataset: {}, <br>Quantiles: {}, Group Id: {}'.format(forecast_cfg['forecast'],
                                                                          results['experiment_cfg']['architecture'],
                                                                          results['experiment_cfg']['vars_definition'],
                                                                          results['experiment_cfg']['dataset_config'],
                                                                          results['quantiles'],
                                                                          id)
        if 'objective_space' in results:
            obj = np.round(results['objective_space'], 3)
            title += '<br>Objective space: {}'.format(obj)

        plot_forecast_intervals(forecasts_grouped, n_output_steps, id,
                                markersize=3, mode='light',
                                fill_max_opacity=0.2,
                                additional_vars=['ESc'],
                                additional_rows=[0],
                                additional_data=mkt_data,
                                title=title if general_cfg['plot_title'] else None,
                                save=general_cfg['save_plot'],
                                file_path=os.path.join(img_path, filename+'_id{}'.format(id)),
                                y_range=plot_segment['y_range'],
                                x_range=plot_segment['x_range'],
                                save_png=True,
                                label_scale=1.2,
                                size=(1980, 1080*2//3),
                                )
