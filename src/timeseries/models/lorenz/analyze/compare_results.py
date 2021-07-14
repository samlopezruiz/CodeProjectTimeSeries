import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.utils.files import save_csv

from timeseries.models.utils.results import load_results
from timeseries.plotly.plot import plot_multiple_results

if __name__ == '__main__':
    # %%
    res_cfg = {'preffix': 'DCNN_WAVENET_CNN_CNN-LSTM_ConvLSTM_STROGANOFF_GP-REGRESS_ARIMA',
               'date': '2021_06_11', 'save_results': True, 'steps': 1,
               'plot_title': False, 'model': 'lorenz', 'stage': 'comparison',
               'image_folder': 'images', 'results_folder': 'results',
               'series': 'trend_noise_s'}

    compare = ('variate', ['uni', 'multi'])
    var = ('n params', None)
    # drop = ['ENSEMBLE', 'ARIMA', 'GP-REGRESS', 'STROGANOFF']
    drop = ['ENSEMBLE', 'ARIMA', 'WAVENET', 'CNN', 'CNN-LSTM', 'DCNN' , 'ConvLSTM']

    plot_multiple_results(res_cfg, compare, var, size=(1980 // 2, 1080), label_scale=1.5, drop=drop)

    try:
        (in_cfg, input_cfg, names, model_cfgs, summary), file_name = load_results(res_cfg)
        save_csv(summary, res_cfg)
    except:
        pass
