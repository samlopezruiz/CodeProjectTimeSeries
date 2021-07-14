import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.plotly.plot import plot_multiple_results
from timeseries.utils.files import save_csv

if __name__ == '__main__':
    #%%
    res_cfg = {'preffix': 'DCNN_WAVENET_CNN_CNN-LSTM_ConvLSTM_STROGANOFF_GP-REGRESS_ENSEMBLE',
               'date': '2021_06_11', 'save_results': True, 'steps': None,
               'plot_title': False, 'model': 'lorenz', 'stage': 'comparison',
               'image_folder': 'images', 'results_folder': 'results',
               'series': 'trend_noise_s'}

    compare = ('steps', [1, 3, 6])
    var = ('n params', None)
    drop = ['ENSEMBLE', 'GP-REGRESS', 'STROGANOFF']
    # drop = ['ENSEMBLE', 'WAVENET', 'CNN', 'CNN-LSTM', 'DCNN', 'ConvLSTM']
    data, errors = plot_multiple_results(res_cfg, compare, var, size=(1980//2, 1080), label_scale=1.5, drop=drop)
    save_csv(data, res_cfg, suffix='data')
    save_csv(errors, res_cfg, suffix='errors')





