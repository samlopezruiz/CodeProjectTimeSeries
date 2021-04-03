from timeseries.models.lorenz.univariate.onestep.arima.func import arima_forecast, arima_creation
import pandas as pd
from timeseries.models.lorenz.functions.explore import explore_model


if __name__ == '__main__':
    cfg = (9, 1, 5)
    metrics1 = explore_model(arima_forecast, arima_creation, cfg, granularity=1, name="grain=1")
    metrics2 = explore_model(arima_forecast, arima_creation, cfg, end_time=200, name="et=200")
    metrics3 = explore_model(arima_forecast, arima_creation, cfg, granularity=5, end_time=100, name='grain=5/et=100')

#%%
    print(pd.concat([metrics2, metrics1, metrics3], axis=1))
    metrics_end_time = pd.concat([metrics1, metrics2], axis=1)
    metrics_end_time["comp"] = 1 - metrics_end_time["grain=1"] / metrics_end_time["et=200"]
    print(metrics_end_time)