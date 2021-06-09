
import matplotlib.pyplot as plt
import pmdarima as pm

from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.models.lorenz.univariate.onestep.arima.func import arima_get_one_step_uv_funcs

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name = "ARIMA"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, 'order': (9, 1, 5)}
    functions = arima_get_one_step_uv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)


    # Seasonal - fit stepwise auto-ARIMA
    smodel = pm.auto_arima(train_pp, start_p=3, start_q=3, d=0,
                           test='adf',
                           max_p=12, max_q=12,
                           start_P=0, seasonal=False, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

    print(smodel.summary())

    smodel.plot_diagnostics(figsize=(7, 5))
    plt.show()