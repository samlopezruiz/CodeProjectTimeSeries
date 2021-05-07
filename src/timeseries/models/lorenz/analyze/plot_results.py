from timeseries.plotly.plot import plot_multiple_results, plot_multiple_results_2rows

if __name__ == '__main__':
    res_cfg = {'suffix': 'DCNN_WAVENET_CNN_CNN-LSTM_ConvLSTM_ENSEMBLE',
               'date': '2021_05_07', 'save_results': True, 'steps': None,
               'plot_title': True, 'model': 'lorenz', 'stage': 'comparison',
               'image_folder': 'images', 'results_folder': 'results',
               'series': 'trend_noise_s'}

    var = ('n params', None)
    steps = [1, 3, 6]
    plot_multiple_results_2rows(res_cfg, steps, var)






