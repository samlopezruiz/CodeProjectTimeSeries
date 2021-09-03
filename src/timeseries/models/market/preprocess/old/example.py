import pandas as pd
from sklearn.preprocessing import StandardScaler

from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.market.utils.preprocessing import reconstruct_pred
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series
from timeseries.preprocessing.func import ln_returns

if __name__ == '__main__':
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)

    lorenz_df['x_r'] = ln_returns(lorenz_df['x'])

    i = 0
    test_pp = lorenz_df['x_r'].to_numpy()
    unscaled_y = lorenz_df['x'].to_numpy()
    model_n_steps_out = 6

    ss = StandardScaler()
    scaled_pred_y = ss.fit_transform(lorenz_df['x_r'].to_numpy().reshape(-1, 1))

    forecast_reconst = reconstruct_pred(scaled_pred_y, model_n_steps_out, unscaled_y, ss=ss)
    metrics = forecast_accuracy(forecast_reconst, unscaled_y)
    print(metrics)

    #%%
    df = pd.DataFrame()
    df['forecast_reconst'] = forecast_reconst
    df['y_true'] = unscaled_y
    plotly_time_series(df)

