import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.utils.metrics import get_data_error
from timeseries.experiments.utils.results import load_results
from timeseries.plotly.plot import plot_multiple_results, plot_multiple_results, plot_bar_summary, drop_rows
from timeseries.utils.files import load_data_err

#%%

if __name__ == '__main__':
    #%%
    res_cfg = {'preffix': 'DCNN_WAVENET_CNN_CNN-LSTM_ConvLSTM_STROGANOFF_GP-REGRESS_ARIMA',
               'date': '2021_06_11', 'save_results': False, 'steps': 1,
               'plot_title': False, 'model': 'lorenz', 'stage': 'univariate',
               'image_folder': 'res_img', 'results_folder': 'results',
               'series': 'trend_noise_s', 'score_type': 'minmax'}
    # drop = ['ENSEMBLE', 'WAVENET', 'CNN', 'CNN-LSTM', 'DCNN', 'ConvLSTM']
    drop = ['ENSEMBLE', 'GP-REGRESS', 'STROGANOFF']
    (in_cfg, input_cfg, names, model_cfgs, summary), file_name = load_results(res_cfg)
    drop_rows(summary, drop)

    data, errors = get_data_error(summary, res_cfg['score_type'], overall=False, ms=False)
    cfg = {'n_steps_out': in_cfg['steps'], 'n_series': in_cfg['n_series'], 'n_repeats': in_cfg['n_repeats']}
    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'CONFIG: ' + str(cfg),
                     file_path=[res_cfg['image_folder'], file_name], plot_title=res_cfg['plot_title'], showlegend=False,
                     save=res_cfg['save_results'], n_cols_adj_range=data.shape[1], size=(1980, 1080//2), label_scale=1.5)






