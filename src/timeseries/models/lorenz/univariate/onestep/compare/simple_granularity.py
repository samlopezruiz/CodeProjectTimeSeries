from timeseries.models.lorenz.functions.explore import explore_model
from timeseries.models.lorenz.univariate.onestep.simple.func import simple_forecast, simple_fit
import pandas as pd

if __name__ == '__main__':
    cfg = (1, 1, "persist")
    metrics1 = explore_model(simple_forecast, simple_fit, cfg, granularity=1, name="grain=1")
    metrics2 = explore_model(simple_forecast, simple_fit, cfg, end_time=200, name="et=200")
    metrics3 = explore_model(simple_forecast, simple_fit, cfg, granularity=5, end_time=100, name='grain=5/et=100')

#%%
    print(pd.concat([metrics2, metrics1, metrics3], axis=1))
    metrics_end_time = pd.concat([metrics1, metrics2], axis=1)
    metrics_end_time["comp"] = 1 - metrics_end_time["grain=1"] / metrics_end_time["et=200"]
    print(metrics_end_time)