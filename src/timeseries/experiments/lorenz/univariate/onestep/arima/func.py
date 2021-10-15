import time

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_forecast(model, steps=1,  history=None, cfg=None):
    yhat = model.forecast(steps=steps)
    if steps == 1:
        return yhat[0]
    else:
        return yhat

def arima_creation(train, cfg):
    order = cfg['order']
    model = SARIMAX(endog=train, order=order, seasonal_order=(0, 0, 0, 0), trend='n',
                    enforce_stationarity=False)
    return model.fit(disp=0)


def arima_one_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    order = cfg['order']
    model = SARIMAX(endog=train, order=order, seasonal_order=(0, 0, 0, 0), trend='n',
                    enforce_stationarity=False)
    start_time = time.time()
    model_fitted = model.fit(disp=0)
    train_time = round((time.time() - start_time), 2)
    loss = 0
    return model_fitted, train_time, loss


def arima_one_step_uv_predict(model, history, cfg, steps=1):
    updated_model = model.append(np.array([history[-1]]))
    yhat = model.forecast(steps=steps)
    if steps == 1:
        return np.array([yhat[0]]), updated_model
    else:
        return yhat, updated_model

def arima_get_one_step_uv_funcs():
    return [arima_one_step_uv_predict, arima_one_step_uv_fit]
