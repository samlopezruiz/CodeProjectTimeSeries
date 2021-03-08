from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np


# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
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


if __name__ == '__main__':
    from timeseries.data.lorenz.lorenz import Lorenz

    # define dataset
    # data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    start_time = 0
    end_time = 100
    t = np.linspace(start_time, end_time, end_time * 100)

    lorenz_sys = Lorenz(sigma=10., rho=28., beta=8. / 3.)
    lorenz_sys.solve(t)
    xyz = lorenz_sys.get_time_series()
    data = xyz[0]
    print(len(data))
    # data split
    n_test = 200
    # model configs
    cfg_list = sarima_configs()
    # grid search
    # scores = grid_search(data, cfg_list, n_test, parallel=False)
    # print('done')
    # # list top 3 configs
    # for cfg, error in scores[:3]:
    #     print(cfg, error)
