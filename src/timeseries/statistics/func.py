import numpy as np
from hurst import compute_Hc
from numpy import sqrt, std, subtract, polyfit, log
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import shapiro, normaltest
import pmdarima as pm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

from timeseries.statistics.plot import plot_correlogram


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


def normality_test(x, name='', print_=True):
    x = x[~np.isnan(x)]
    sh = shapiro(x)
    if print_:
        print('\n-----{} NORMALITY TEST-----'.format(name))
        print('Shapiro Test: w={}, p-value={}'.format(sh[0], sh[1]))
        print('If p-value < 0.05, null hypothesis is rejected and the data normal')
    return sh


def correlation_test(x, lags=None, print_=True, name=''):
    x = x[~np.isnan(x)]
    lags = min(10, len(x) // 5) if lags is None else lags
    # q_p = np.max(q_stat(acf(x, nlags=lags, fft=True), len(x))[1])
    q_p = np.max(acorr_ljungbox(x, lags=lags, return_df=False)[1])
    if print_:
        print('\n-----{} CORRELATION TEST-----'.format(name))
        print('Box-Ljung Test: p-value={}'.format(q_p))
        print('If p-value > 0.05, null hypothesis si rejected and the data is uncorrelated')
    return q_p


def stationary_test(x, name=''):
    x = x[~np.isnan(x)]
    adf = adfuller(x)
    print('\n-----{} STATIONARITY TEST-----'.format(name))
    print('ADF: p-value={}'.format(adf[1]))
    print('If p-value > 0.05, null hypothesis is rejected and the data is non-stationary')
    return adf


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def stat_test(x, name, file_name='plot', save=False):
    plot_correlogram(x, title=name, file_name=file_name+' '+name, save=save)
    normality_test(x, name=name)
    correlation_test(x, name=name)
    stationary_test(x, name=name)


def arima_garch(returns, arima_order, garch_order=(1, 1), dataset_name='', save_results=False):
    arima_model = ARIMA(endog=returns, order=arima_order).fit()
    print(arima_model.summary())
    arima_resid = arima_model.resid
    arima_resid = arima_resid[~np.isnan(arima_resid)]
    arima_volatility = np.power(arima_resid, 2)

    name = 'ARIMA RESIDUALS'
    stat_test(arima_resid, name, file_name=dataset_name, save=save_results)

    name = 'ARIMA VOLATILITY'
    stat_test(arima_volatility, name, file_name=dataset_name, save=save_results)

    # %%
    archmodel = arch_model(arima_resid, mean='ZERO', vol='GARCH', q=garch_order[0], p=garch_order[1])
    model_fit = archmodel.fit(disp='off')
    arch_resid = model_fit.resid / model_fit.conditional_volatility
    arch_vol = model_fit.conditional_volatility
    arch_volatility = np.power(arch_resid, 2)
    print(model_fit.summary())

    name = 'ARCH RESIDUALS'
    stat_test(arch_resid, name, file_name=dataset_name, save=save_results)
    name = 'ARCH VOLATITLITY'
    stat_test(arch_volatility, name, file_name=dataset_name, save=save_results)

    name = dataset_name + ' VOLATILITY'
    plot_volatility_returns(returns, arch_vol, file_name=name, save=save_results)


def coeff_garch(returns, garch_order=(1, 1)):
    ss = StandardScaler()
    norm_returns = ss.fit_transform(returns.reshape(-1, 1))
    archmodel = arch_model(norm_returns, mean='ZERO', vol='GARCH', q=garch_order[0], p=garch_order[1])
    res = archmodel.fit(disp='off')
    arch_vol = res.conditional_volatility
    params = res.params
    return params, arch_vol


def stats_kpi(price_hist, returns, arima_order=(5, 0, 5)):
    hurst_res = compute_Hc(price_hist, kind='price', simplified=True)
    bjl = correlation_test(returns, print_=False)
    garch_params, arch_vol = coeff_garch(returns)
    std_price, std_returns = np.std(price_hist), np.std(returns)
    # arima_model = ARIMA(endog=returns, order=arima_order).fit()
    norm = normaltest(returns)
    return hurst_res[0], bjl, garch_params, arch_vol, std_price, std_returns,  norm[1]
