import numpy as np
import pandas as pd
from hmmlearn import hmm

from algorithms.hmm.func import fitHMM
from timeseries.data.lorenz.lorenz import lorenz_system
from timeseries.plotly.plot import plotly_time_series
from timeseries.preprocessing.func import ln_returns


def regime_lorenz_system():
    df1, xyz, t, lorenz_sys1 = lorenz_system(end_time=130, granularity=5, positive_offset=True,
                                             noise=False, sigma=1, trend=False, beta=8. / 3.)
    df2, xyz, t, lorenz_sys2 = lorenz_system(end_time=130, granularity=5, positive_offset=True,
                                             noise=False, sigma=1, trend=False, beta=1. / 3.)

    Z = ((np.array((range(df1.shape[0]))) % 500) > 250).astype(int)
    df = pd.DataFrame()
    vars = ['x', 'y', 'z']
    for var in vars:
        df[var] = (Z == 0) * df1[var] + (Z == 1) * df2[var]
    X = df.loc[:, vars].to_numpy()
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=2)
    df['hmm'] = hidden_states

    return df, hidden_proba


if __name__ == '__main__':
    save_folder = 'images'
    save_plots = False
    plot_titles = True
    df, hidden_proba = regime_lorenz_system()
    plotly_time_series(df, features=['x', 'y', 'z', 'hmm'],
                       rows=list(range(4)), markers='lines')
    # df1, xyz, t, lorenz_sys1 = lorenz_system(end_time=130, granularity=5, positive_offset=True,
    #                                          noise=False, sigma=1, trend=False, beta=8. / 3.)
    # df2, xyz, t, lorenz_sys2 = lorenz_system(end_time=130, granularity=5, positive_offset=True,
    #                                          noise=False, sigma=1, trend=False, beta=1. / 3.)
    # # lorenz_sys1.plot_time_series(file_path=[save_folder, 'lorenz-attractor-time-series'], save=save_plots,
    # #                              title_bool=plot_titles)
    # # lorenz_sys2.plot_time_series(file_path=[save_folder, 'lorenz-attractor-time-series'], save=save_plots,
    # #                              title_bool=plot_titles)
    #
    # x = np.array((range(df1.shape[0])))
    # Z = ((x % 500) > 250).astype(int)
    #
    # df = pd.DataFrame()
    # for var in ['x', 'y', 'z']:
    #     df[var] = (Z == 0) * df1[var] + (Z == 1) * df2[var] #+ (Z == 2) * df3[var]
    #     # df[var + '_r'] = ln_returns(df[var])
    # df['state'] = Z
    # # df.dropna(inplace=True)
    # vars = ['x', 'y', 'z']
    # X = df.loc[:, vars].to_numpy()
    # hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=2)
    # df['hmm'] = hidden_states

    plotly_time_series(df, features=['x', 'y', 'z', 'state', 'hmm'],
                       rows=list(range(5)), markers='lines')
