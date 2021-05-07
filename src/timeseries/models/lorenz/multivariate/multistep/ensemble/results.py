import os
import joblib
from timeseries.models.utils.results import load_results


if __name__ == '__main__':
    res_cfg = {'steps': 6, 'date': '2021_05_06', 'save_results': True,
               'plot_title': True, 'stage': 'comparison',
               'image_folder': 'images', 'results_folder': 'results',
               'suffix': 'DCNN_WAVENET_CNN_CNN-LSTM_ConvLSTM',
               'series': 'trend_noise_s', 'model': 'lorenz', 'series': 'trend_noise_s'}

    # in_cfg, input_cfg, names, model_cfgs, summary = load_results(res_cfg)
    file_name = 'ensemble_s6_2021_05_06'
    in_cfg, input_cfg, names, model_cfgs, summary = joblib.load(os.path.join('results', file_name+'.z'))
    summary_sorted = summary.sort_values(['score_m'], ascending=False)
