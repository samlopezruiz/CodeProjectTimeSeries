from timeseries.plotly.plot import plot_multiple_results, plot_multiple_results

if __name__ == '__main__':
    res_cfg = {'preffix': 'DCNN_WAVENET_CNN_CNN-LSTM_ConvLSTM_ENSEMBLE',
               'date': '2021_06_08', 'save_results': False, 'steps': None,
               'plot_title': False, 'model': 'lorenz', 'stage': 'comparison',
               'image_folder': 'images', 'results_folder': 'results',
               'series': 'trend_noise_s'}

    compare = ('steps', [1, 3, 6])
    var = ('n params', None)
    data, errors = plot_multiple_results(res_cfg, compare, var, size=(1980//2, 1080), label_scale=1.5)






