from timeseries.plotly.plot import plot_multiple_results

if __name__ == '__main__':
    #%%
    res_cfg = {'preffix': 'DCNN_WAVENET_CNN_CNN-LSTM_ConvLSTM',
               'date': '2021_06_08', 'save_results': True, 'steps': 1,
               'plot_title': False, 'model': 'lorenz', 'stage': 'comparison',
               'image_folder': 'images', 'results_folder': 'results',
               'series': 'trend_noise_s'}

    compare = ('variate', ['uni', 'multi'])
    var = ('n params', None)

    plot_multiple_results(res_cfg, compare, var, size=(1980//2, 1080), label_scale=1.5)





