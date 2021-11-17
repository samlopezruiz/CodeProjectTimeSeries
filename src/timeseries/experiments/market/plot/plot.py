import numpy as np
from matplotlib import pyplot as plt

from algorithms.moo.utils.plot import save_fig
from timeseries.experiments.market.split.func import merge_train_test_groups
from timeseries.plotly.plot import plotly_ts_candles, plotly_ts_regime
import seaborn as sns
sns_colors = sns.color_palette()

def plot_mkt_candles(df, inst, features=None, resample=False, period='90T', ts_height=0.6, template='plotly_white'):
    df_plot = df.resample(period).last() if resample else df
    features = list(df.columns[4:]) if features is None else features
    plotly_ts_candles(df_plot, features=features, instrument=inst, adjust_height=(True, ts_height),
                      template=template, rows=[i for i in range(len(features))])


def plot_train_test_groups(df, split_cfg=None, plot_last=None, regime_col='test', features=['ESc', 'subset'],
                           resample=False, period='90T', template='plotly_dark', save=False, file_path=None):
    # df_merged = merge_train_test_groups(dfs_train, dfs_test)
    if plot_last is not None:
        df = df.iloc[-plot_last:, :]
    df_plot = df.resample(period).last() if resample else df
    title = 'SPLIT CFG: {}'.format(str(split_cfg)) if split_cfg is not None else "SPLIT GROUPS"
    plotly_ts_regime(df_plot, features=features, regime_col=regime_col, title=title, adjust_height=(True, 0.8),
                     template=template, rows=[i for i in range(len(features))], save=save, file_path=file_path)

def get_legend_labels(prefix, suffix=None, length=None):
    if suffix is None and length is None:
        raise ValueError('specify suffix or length')
    if suffix is None:
        labels = ['{} {}'.format(prefix, i) for i in range(length)]
    else:
        labels = ['{} {}'.format(prefix, suffix) for suffix in suffix]

    return labels

def plot_2D_pareto_front(Fs,
                         save=False,
                         file_path=None,
                         selected_ixs=None,
                         original_ixs=None,
                         figsize=(15, 15),
                         use_date=False,
                         xlabel='Quantile coverage probability',
                         ylabel='Quantile estimation error',
                         title='Multi objective optimization',
                         legend_labels_suffix=None,
                         plot_total_loss_sublot=True):
    fig, ax = plt.subplots(2 if plot_total_loss_sublot else 1, 1, figsize=figsize)
    ax = ax if plot_total_loss_sublot else [ax]



    if isinstance(Fs, list):
        labels = get_legend_labels('Pareto front', suffix=legend_labels_suffix, length=len(Fs))

        for i, F in enumerate(Fs):
            ax[0].plot(F[:, 0], F[:, 1],
                       'o',
                       markersize=8,
                       label=labels[i])
    else:
        ax[0].plot(Fs[:, 0], Fs[:, 1],
                   'o',
                   markersize=8,
                   label='Pareto front')

    if original_ixs is not None:
        if isinstance(original_ixs, list):
            for i, ix, F in enumerate(zip(original_ixs, Fs)):
                ax[0].plot(F[ix, 0], F[ix, 1],
                           'k*',
                           markersize=24,
                           label='Original solution' if i == 0 else None)
        else:
            ax[0].plot(Fs[original_ixs, 0], Fs[original_ixs, 1],
                       'k*',
                       markersize=24,
                       label='Original solution')

    if selected_ixs is not None:
        if isinstance(Fs, list):
            for i, F in enumerate(Fs):
                ax[0].plot(F[selected_ixs, 0], F[selected_ixs, 1],
                           '*',
                           markersize=24,
                           color='red',
                           label='Selected solution' if i == 0 else None)
        else:
            ax[0].plot(Fs[selected_ixs, 0], Fs[selected_ixs, 1],
                       '*',
                       markersize=24,
                       color='red',
                       label='Selected solution')
            if plot_total_loss_sublot:
                ax[1].plot(Fs[selected_ixs, 0], Fs[selected_ixs, 1],
                           '*',
                           markersize=24,
                           color='red',
                           label='Selected solution')


    if plot_total_loss_sublot:
        if isinstance(Fs, list):
            for i, F in enumerate(Fs):
                ax[1].plot(F[:, 0], np.sum(F, axis=1),
                           'o',
                           markersize=8,
                           color='gray',
                           label='Total loss' if i == 0 else None)
        else:
            ax[1].plot(Fs[:, 0], np.sum(Fs, axis=1),
                       'o',
                       markersize=8,
                       color='gray',
                       label='Total loss')

        ax[1].set_title('Total Loss', fontweight="bold")
        ax[1].set_xlabel('Solution')
        ax[1].set_ylabel('Loss')
        ax[0].legend()

    ax[0].set_title('Pareto front', fontweight="bold")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].legend()
    fig.suptitle(title)
    plt.tight_layout()
    # plt.legend()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)



def plot_2D_moo_results(Fs, eq_Fs,
                        save=False,
                        file_path=None,
                        selected_ix=None,
                        original_ixs=None,
                        figsize=(15, 15),
                        use_date=False,
                        xlabel='Quantile coverage probability error',
                        ylabel='Quantile estimation error',
                        title='Multi objective optimization',
                        legend_labels_suffix=None,
                        xaxis_limit=None):
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    if xaxis_limit is not None:
        if isinstance(Fs, list):
            Fs_x_plot_masks = [F[:, 0] < xaxis_limit for F in Fs]
        else:
            Fs_x_plot_masks = Fs[:, 0] < xaxis_limit
    else:
        if isinstance(Fs, list):
            Fs_x_plot_masks = [np.ones((F.shape[0],)).astype(bool) < xaxis_limit for F in Fs]
        else:
            Fs_x_plot_masks = np.ones((Fs.shape[0],)).astype(bool)


    if isinstance(Fs, list):
        labels = get_legend_labels('Pareto front', suffix=legend_labels_suffix, length=len(Fs))

        for i, F in enumerate(Fs):
            ax[0, 0].plot(F[Fs_x_plot_masks[i], 0], F[Fs_x_plot_masks[i], 1],
                          'o',
                          markersize=8,
                          color=sns_colors[i],
                          label=labels[i])
    else:
        ax[0, 0].plot(Fs[Fs_x_plot_masks, 0], Fs[Fs_x_plot_masks, 1],
                      'o',
                      markersize=8,
                      color=sns_colors[0],
                      label='Pareto front')

    if isinstance(Fs, list):
        for i, F in enumerate(Fs):
            labels = get_legend_labels('Error', suffix=legend_labels_suffix, length=len(Fs))
            ax[0, 1].plot(F[Fs_x_plot_masks[i], 0], np.sum(F[Fs_x_plot_masks[i], :], axis=1),
                          'o',
                          markersize=8,
                          color=sns_colors[i],
                          label=labels[i])
    else:
        ax[0, 1].plot(Fs[Fs_x_plot_masks, 0], np.sum(Fs[Fs_x_plot_masks, :], axis=1),
                      'o',
                      markersize=8,
                      color='gray',
                      label='Error')

    if isinstance(eq_Fs, list):
        QCP_labels = get_legend_labels('QCP', suffix=legend_labels_suffix, length=len(Fs))
        QEE_labels = get_legend_labels('QEE', suffix=legend_labels_suffix, length=len(Fs))
        for i, (F, eq_F) in enumerate(zip(Fs, eq_Fs)):
            ax[1, 0].plot(F[Fs_x_plot_masks[i], 0], eq_F[Fs_x_plot_masks[i], 0],
                          'v',
                          markersize=8,
                          color=sns_colors[i],
                          label=QCP_labels[i])
            ax[1, 0].plot(F[Fs_x_plot_masks[i], 0], eq_F[Fs_x_plot_masks[i], 1],
                          '^',
                          markersize=8,
                          color=sns_colors[i],
                          label=QEE_labels[i])
    else:
        ax[1, 0].plot(Fs[Fs_x_plot_masks, 0], eq_Fs[Fs_x_plot_masks, 0],
                      'v',
                      markersize=8,
                      color=sns_colors[0],
                      label='QCP - equal weights')
        ax[1, 0].plot(Fs[Fs_x_plot_masks, 0], eq_Fs[Fs_x_plot_masks, 1],
                      '^',
                      markersize=8,
                      color=sns_colors[0],
                      label='QEE - equal weights')

    if isinstance(eq_Fs, list):
        labels = get_legend_labels('Error', suffix=legend_labels_suffix, length=len(Fs))
        for i, (F, eq_F) in enumerate(zip(Fs, eq_Fs)):
            ax[1, 1].plot(F[Fs_x_plot_masks[i], 0], np.sum(eq_F[Fs_x_plot_masks[i], :], axis=1),
                          'o',
                          markersize=8,
                          color=sns_colors[i],
                          label=labels[i])
    else:
        ax[1, 1].plot(Fs[Fs_x_plot_masks, 0], np.sum(eq_Fs[Fs_x_plot_masks, :], axis=1),
                      'o',
                      markersize=8,
                      color='gray',
                      label='Error')

    if original_ixs is not None:
        highlight_point(Fs, eq_Fs, ax, original_ixs, color='black', label='Original Solution')
    if selected_ix is not None:
        highlight_point(Fs, eq_Fs, ax, selected_ix, color='red', label='Selected Solution')

    ax[0, 1].set_title('Total Error', fontweight="bold")
    ax[0, 1].set_xlabel('Quantile coverage probability error')
    ax[0, 1].set_ylabel('Error')
    ax[0, 1].legend()

    ax[0, 0].set_title('Pareto front', fontweight="bold")
    ax[0, 0].set_xlabel(xlabel)
    ax[0, 0].set_ylabel(ylabel)
    ax[0, 0].legend()

    ax[1, 1].set_title('Total Error - equal weights', fontweight="bold")
    ax[1, 1].set_xlabel('Quantile coverage probability error')
    ax[1, 1].set_ylabel('Error')
    ax[1, 1].legend()

    ax[1, 0].set_title('Error - equal weights', fontweight="bold")
    ax[1, 0].set_xlabel('Quantile coverage probability error')
    ax[1, 0].set_ylabel('Error')
    ax[1, 0].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def highlight_point(Fs, eq_Fs, ax, ixs, color, label):
    if isinstance(Fs, list) and isinstance(eq_Fs, list):
        for i, (F, eq_F) in enumerate(zip(Fs, eq_Fs)):
            ax[0, 0].plot(F[ixs[i], 0], F[ixs[i], 1],
                          '*',
                          markersize=24,
                          color=color,
                          label=label if i == 0 else None)
            ax[0, 1].plot(F[ixs[i], 0], np.sum(F, axis=1)[ixs[i]],
                          '*',
                          markersize=24,
                          color=color,
                          label=label if i == 0 else None)
            ax[1, 1].plot(F[ixs[i], 0], np.sum(eq_F, axis=1)[ixs[i]],
                          '*',
                          markersize=24,
                          color=color,
                          label=label if i == 0 else None)
            ax[1, 0].plot(F[ixs[i], 0], eq_Fs[ixs[i], 0],
                          'v',
                          markersize=8,
                          color=color,
                          label=label)
            ax[1, 0].plot(F[ixs[i], 0], eq_Fs[ixs[i], 1],
                          '^',
                          markersize=8,
                          color=color,
                          label=label)

    else:
        ax[0, 0].plot(Fs[ixs, 0], Fs[ixs, 1],
                      '*',
                      markersize=24,
                      color=color,
                      label=label)
        ax[0, 1].plot(Fs[ixs, 0], np.sum(Fs, axis=1)[ixs],
                      '*',
                      markersize=24,
                      color=color,
                      label=label)
        ax[1, 1].plot(Fs[ixs, 0], np.sum(eq_Fs, axis=1)[ixs],
                      '*',
                      markersize=24,
                      color=color,
                      label=label)
        ax[1, 0].plot(Fs[ixs, 0], eq_Fs[ixs, 0],
                      'v',
                      markersize=8,
                      color=color,
                      label=label)
        ax[1, 0].plot(Fs[ixs, 0], eq_Fs[ixs, 1],
                      '^',
                      markersize=8,
                      color=color,
                      label=label)
