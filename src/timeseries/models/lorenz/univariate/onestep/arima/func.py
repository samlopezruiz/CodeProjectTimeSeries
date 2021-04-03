from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_forecast(model, steps=1,  history=None, cfg=None):
    yhat = model.forecast(steps=steps)
    if steps == 1:
        return yhat[0]
    else:
        return yhat


def arima_creation(train, config):
    model = SARIMAX(endog=train, order=config, seasonal_order=(0, 0, 0, 0), trend='n',
                    enforce_stationarity=False)
    return model.fit(disp=0)

