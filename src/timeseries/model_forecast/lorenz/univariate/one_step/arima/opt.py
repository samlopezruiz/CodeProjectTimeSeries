
import pmdarima as pm
import numpy as np
from timeseries.data.lorenz.lorenz_wrapper import lorenz_system
from timeseries.model.univariate.explore.functions import train_test_split
import matplotlib.pyplot as plt


if __name__ == '__main__':
    lorenz_df, xyz, t = lorenz_system()
    test_size = 1000
    x = lorenz_df['x']
    x = x[x.index > 15]
    data = np.array(x)
    train, test = train_test_split(data, test_size)

    # Seasonal - fit stepwise auto-ARIMA
    smodel = pm.auto_arima(train, start_p=3, start_q=3, d=1,
                           test='adf',
                           max_p=8, max_q=8,
                           start_P=0, seasonal=False, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

    print(smodel.summary())

    smodel.plot_diagnostics(figsize=(7, 5))
    plt.show()