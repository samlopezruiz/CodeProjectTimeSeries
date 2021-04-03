from timeseries.data.lorenz.lorenz import lorenz_system
from timeseries.models.lorenz.univariate.onestep.arima.func import arima_creation, arima_forecast
from timeseries.models.lorenz.functions.functions import train_test_split
from timeseries.models.lorenz.univariate.onestep.simple.func import simple_fit, simple_forecast
from timeseries.models.utils.metrics import compare_forecast


if __name__ == '__main__':
    lorenz_df, xyz, t, _ = lorenz_system()
    test_size = 1000
    data = xyz[0]

    models = {'simple': (simple_fit, simple_forecast, (1, 1, 'persist')),
              'arima': (arima_creation, arima_forecast, (9, 1, 5))}

    train, test = train_test_split(data, test_size)
    t_train, t_test = train_test_split(t, test_size)
    metrics = compare_forecast(models, train, test)
