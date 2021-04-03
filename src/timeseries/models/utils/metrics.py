import numpy as np
from statsmodels.tsa.stattools import acf
import pandas as pd
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast


def forecast_accuracy(forecast, test):
    omin = -np.min(np.hstack([forecast, test]))
    forecast = np.array(forecast) + omin + 1
    test = np.array(test) + omin + 1
    mape = np.mean(np.abs(forecast - test) / np.abs(test))  # MAPE
    me = np.mean(forecast - test)             # ME
    mae = np.mean(np.abs(forecast - test))    # MAE
    mpe = np.mean((forecast - test) / test)   # MPE
    rmse = np.mean((forecast - test) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, test)[0, 1]   # corr
    mins = np.amin(np.vstack([forecast, test]), axis=0)
    maxs = np.amax(np.vstack([forecast, test]), axis=0)
    minmax = np.mean(mins / maxs)
    acf1 = acf(forecast - test, fft=False)[1]                      # ACF1
    return({'mape': round(mape, 4), 'me': round(me,4), 'mae': round(mae,4),
            'mpe': round(mpe,4), 'rmse': round(rmse,4), 'acf1': round(acf1,4),
            'corr': round(corr,4), 'minmax': round(minmax,4)})


def compare_forecast(models, train, test):
    metrics = []
    for model_name, (creation, forecast, cfg) in models.items():
        print("Forecast with {}".format(model_name))
        error, forecast = walk_forward_step_forecast(train, test, cfg, forecast, creation)
        metrics.append(forecast_accuracy(forecast, test))

    df = pd.DataFrame(metrics).transpose()
    df.columns = models.keys()
    return df