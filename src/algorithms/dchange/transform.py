from timeseries.data.lorenz.lorenz import multivariate_lorenz
from algorithms.dchange.func import direct_change
from timeseries.plotly.plot import plotly_time_series
import pandas as pd


if __name__ == '__main__':
    # %% INPUTS
    save_folder = 'images'
    save_plots = False
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=1, noise=True,
                                                                  positive_offset=True, trend=True)
    name = "DIRECTIONAL CHANGE"
    x = lorenz_df['x']

    # %% TRANSFORM
    tholds = [0.07, 0.05, 0.03]
    dc_cfg = {'thold': 0.05, 'delta_t': 0.01}
    # tholds = [0.05]
    dc_dfs = []
    for thold in tholds:
        dc_cfg['thold'] = thold
        dc_dfs.append(direct_change(x, dc_cfg))

    #%%
    # plotly_time_series(dc_dfs[0], rows=[0, 1, 2, 3], title=name, markers='lines+markers',
    #                    file_path=[save_folder, name + 'dc_'+str(tholds[0])], save=save_plots)

    # %%
    comb = pd.DataFrame([x]+[dc['dc'] for dc in dc_dfs]).T
    comb.columns = ['x'] + ['dc_' + str(thold) for thold in tholds]

    plotly_time_series(comb[0:100], rows=[0]+[1]*len(tholds), title=name, markers='lines+markers',
                       file_path=[save_folder, name], save=save_plots, alphas=[0.3]+[0.7]*len(tholds))

    # plotly_time_series(comb[0:10], rows=list(range(len(tholds) + 1)), title=name, markers='lines+markers',
    #                    file_path=[save_folder, name], save=save_plots)


