from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pmdarima as pm



def sarima_forecast(history, config):
    order, sorder, trend = config
    # define models
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit models
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define architectures lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create architectures instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models

def auto_arima(x, max_p=8, max_q=8):
    x = x[~np.isnan(x)]
    smodel = pm.auto_arima(x, start_p=1, start_q=1, d=0,
                           test='adf',
                           max_p=max_p, max_q=max_q,
                           start_P=0, seasonal=False, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

    return smodel

if __name__ == '__main__':
    from timeseries.data.lorenz.lorenz import Lorenz, univariate_lorenz

    lorenz_df, train, test, t_train, t_test = univariate_lorenz()
    # models configs
    cfg_list = sarima_configs()
    # grid search
    scores = grid_search(train, test, cfg_list, sarima_forecast, parallel=True)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
