import warnings

import seaborn as sns
from matplotlib import pyplot as plt

from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.preprocessing.func import ln_returns
from timeseries.statistics.func import stat_test
from timeseries.statistics.plot import plot_correlogram

sns.set_theme()
sns.set_context("poster", font_scale=1)

warnings.filterwarnings("ignore")
plt.rc("figure", figsize=(16, 12))
plt.rc("savefig", dpi=90)
plt.rc("font", family="sans-serif")
plt.rc("font", size=18)


if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'save_results': True}

    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)

    # var = 'x'
    for var in ['x', 'y', 'z']:
        plot_correlogram(ln_returns(lorenz_df[var]), title=None, file_name=var+'_ln_stats', save=in_cfg['save_results'])
        plot_correlogram(lorenz_df[var], title=None, file_name=var + '_stats', save=in_cfg['save_results'])