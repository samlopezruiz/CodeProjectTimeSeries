import datetime
import os

from timeseries.data.lorenz.lorenz import lorenz_system
from timeseries.plotly.plot import plotly_time_series
import numpy as np
import matplotlib.pyplot as plt

from timeseries.utils.files import new_dir

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def mean_diff(df):
    x = df['x']
    x_diff = np.abs(x.diff().dropna())
    return np.mean(x_diff)


def plot_steps_needed(thold, in_out_ratio=3, save=False, file_name='steps'):
    out_steps, in_steps = ([], [])
    granularities = list(range(1, 11))
    for g in granularities:
        lorenz_df, _, _, lorenz_sys = lorenz_system(granularity=g)
        x_mean = mean_diff(lorenz_df)
        out_steps.append(thold / x_mean)
        in_steps.append(thold / x_mean * in_out_ratio)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.plot(granularities, out_steps, 'o-', label='output steps')
    # plt.plot(granularities, in_steps, 'o-', label='input steps')
    for xy in zip(granularities, out_steps):
        ax.annotate('%.0f' % xy[1], xy=xy, xytext=(5, 5), color='gray', textcoords='offset pixels')
    # for xy in zip(granularities, in_steps):
    #     ax.annotate('%.0f' % xy[1], xy=xy, xytext=(5, 5), color='gray', textcoords='offset pixels')
    plt.title(r'$\triangle$p = {}'.format(thold))
    plt.xlabel('period')
    plt.ylabel('steps')
    plt.legend()
    ax.set_xticks(np.arange(1, 10), minor=True)
    plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.2)
    fig.tight_layout()
    if save:
        new_dir('images')
        file_name = file_name + '_' + str(thold) + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M") + ".png"
        plt.savefig(os.path.join('images', file_name))
    plt.show()


import seaborn as sns
sns.set_theme()
sns.set_context("poster", font_scale=1)

if __name__ == '__main__':
    save_folder = 'images'
    save_plots = True
    thold = 15
    in_out_ratio = 3

#%%
    plot_steps_needed(thold=10, save=save_plots)
    # plot_steps_needed(thold=15, save=save_plots)
    # plot_steps_needed(thold=20, save=save_plots)

#%%
#     lorenz_df, _, _, lorenz_sys = lorenz_system(granularity=1)
#     # lorenz_sys.plot3d(file_path=[save_folder, 'lorenz-attractor-3d-grain1'], save=save_plots)
#     # lorenz_sys.plot2d(file_path=[save_folder, 'lorenz-attractor-phase-plane-grain1'], save=save_plots)
#     lorenz_sys.plot_time_series(ini=80, end=90, file_path=[save_folder, 'lorenz-attractor-time-series-grain1'],
#                                 save=save_plots, markers='lines+markers')
#
# #%%
#     lorenz_df, _, _, lorenz_sys = lorenz_system(granularity=5)
#     # lorenz_sys.plot3d(file_path=[save_folder, 'lorenz-attractor-3d-grain5'], save=save_plots)
#     # lorenz_sys.plot2d(file_path=[save_folder, 'lorenz-attractor-phase-plane-grain5'], save=save_plots)
#     lorenz_sys.plot_time_series(ini=80, end=90, file_path=[save_folder, 'lorenz-attractor-time-series-grain5'],
#                                 save=save_plots, markers='lines+markers')
#
# #%%
#     lorenz_df, _, _, lorenz_sys = lorenz_system(granularity=10)
#     # lorenz_sys.plot3d(file_path=[save_folder, 'lorenz-attractor-3d-grain10'], save=save_plots)
#     # lorenz_sys.plot2d(file_path=[save_folder, 'lorenz-attractor-phase-plane-grain10'], save=save_plots)
#     lorenz_sys.plot_time_series(ini=80, end=90, file_path=[save_folder, 'lorenz-attractor-time-series-grain10'],
#                                 save=save_plots, markers='lines+markers')

