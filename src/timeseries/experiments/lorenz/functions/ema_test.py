from algorithms.stroganoff.math import score_error
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.preprocessing.func import ema
from timeseries.plotly.plot import plotly_time_series
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # %% INPUT
    save_folder = 'images'
    plot_title = True
    save_plots = False
    name = "PREPROCESS"
    detrend_ops = ['ln_return', ('ema_diff', 14), 'diff']
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, "positive_offset": True,
                 'detrend': detrend_ops[0], 'trend': True}
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)


    #%%
    x = lorenz_df['x']
    x_ema14 = ema(x, 2)
    df = pd.DataFrame([np.array(x), x_ema14]).T
    df.index = x.index
    plotly_time_series(df, markers='lines',
                       title="SERIES: " + str(input_cfg),
                       file_path=[save_folder, name], plot_title=plot_title, save=save_plots)

    #%%
    print(score_error(x, x_ema14))
