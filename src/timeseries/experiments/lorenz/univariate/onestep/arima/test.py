import os

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.univariate.onestep.arima.func import arima_get_one_step_uv_funcs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.preprocessing import preprocess

#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name = "ARIMA"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, 'order': (6, 0, 2)}
    functions = arima_get_one_step_uv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)
    forecast = np.array([0.3])
    order = model_cfg['order']
    model = SARIMAX(endog=train, order=order, seasonal_order=(0, 0, 0, 0), trend='n',
                    enforce_stationarity=False)
    model_fitted = model.fit(disp=0)

    # forecast_reconst = reconstruct(forecast, train, test, input_cfg, in_cfg['steps'], ss=ss)

    #%%
    import matplotlib.pyplot as plt
    resid = model_fitted.resid[max(order):]
    plt.plot(resid)
    plt.show()
    print(np.mean(np.abs(resid)))