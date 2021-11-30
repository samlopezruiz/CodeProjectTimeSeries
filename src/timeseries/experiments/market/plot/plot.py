import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch

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


def plot_train_test_groups(df,
                           split_cfg=None,
                           plot_last=None,
                           regime_col='test',
                           features=['ESc', 'subset'],
                           resample=False,
                           period='90T',
                           template='plotly_dark',
                           save=False,
                           legend=False,
                           file_path=None,
                           label_scale=1,
                           markersize=5,
                           markers='lines+markers',
                           save_png=False,
                           legend_labels=None
                           ):
    # df_merged = merge_train_test_groups(dfs_train, dfs_test)
    if plot_last is not None:
        df = df.iloc[-plot_last:, :]
    df_plot = df.resample(period).last() if resample else df
    title = 'SPLIT CFG: {}'.format(str(split_cfg)) if split_cfg is not None else "SPLIT GROUPS"
    plotly_ts_regime(df_plot,
                     features=features,
                     regime_col=regime_col,
                     title=title,
                     adjust_height=(True, 0.8),
                     template=template,
                     rows=[i for i in range(len(features))],
                     save=save,
                     markers=markers,
                     file_path=file_path,
                     legend=legend,
                     save_png=save_png,
                     markersize=markersize,
                     label_scale=label_scale,
                     legend_labels=legend_labels)


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
                         plot_total_loss_sublot=True,
                         plot_title=True):
    fig, ax = plt.subplots(2 if plot_total_loss_sublot else 1, 1, figsize=figsize)
    ax = ax if plot_total_loss_sublot else [ax]

    if isinstance(Fs, list):
        labels = get_legend_labels('Model', suffix=legend_labels_suffix, length=len(Fs))

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
    if plot_title:
        fig.suptitle(title)
    plt.tight_layout()
    # plt.legend()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def plot_2D_moo_results_equal_w(Fs,
                                eq_Fs=None,
                                save=False,
                                file_path=None,
                                selected_ixs=None,
                                original_ixs=None,
                                figsize=(15, 15),
                                use_date=False,
                                xlabel='Quantile coverage risk',
                                ylabel='Quantile estimation risk',
                                title='Multi objective optimization',
                                legend_labels=None,
                                plot_title=True,
                                xaxis_limit=None,
                                yaxis_limit_tot_eq=2,
                                markersize=4,
                                add_risk=None):
    if eq_Fs is not None:

        fig = plt.figure(figsize=figsize)
        ax00 = plt.subplot2grid((2, 2), (0, 0))
        ax01 = plt.subplot2grid((2, 2), (0, 1))
        ax10 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        # fig, ax = plt.subplots(2, 2, figsize=figsize)
        # ax00, ax01, ax10, ax11 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    else:
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        ax00, ax01, ax10, ax11 = ax[0], ax[1], ax[0], ax[1]

    axes = [ax00, ax01, ax10]  # , ax11]

    Fs_x_plot_masks = get_x_mask(Fs, xaxis_limit)

    y_min_tot, y_min_tot_eq = 10, 10
    y_max_tot, y_max_tot_eq = 0, 0

    if isinstance(Fs, list):
        # labels = get_legend_labels('Model', suffix=legend_labels, length=len(Fs))

        for i, F in enumerate(Fs):
            if add_risk is not None:
                rect_pareto, rect_total = get_rect_risk(F, add_risk, color_ix=i)
                ax00.add_patch(rect_pareto)
                ax01.add_patch(rect_total)

            ax00.plot(F[:, 0], F[:, 1],
                      '-o',
                      markersize=markersize,
                      color=sns_colors[i],
                      label=legend_labels[i])

            sum_F = np.sum(F[Fs_x_plot_masks[i], :], axis=1)
            y_min_tot = min(y_min_tot, min(sum_F))
            y_max_tot = max(y_max_tot, max(sum_F))
            ax01.plot(F[Fs_x_plot_masks[i], 0], sum_F,
                      '-o',
                      markersize=markersize,
                      color=sns_colors[i],
                      label=legend_labels[i])
    else:
        ax00.plot(Fs[:, 0], Fs[:, 1],
                  '-o',
                  markersize=markersize,
                  color=sns_colors[0],
                  label='Pareto front')

        sum_F = np.sum(Fs[Fs_x_plot_masks], axis=1)
        y_min_tot = min(y_min_tot, min(sum_F))
        y_max_tot = max(y_max_tot, max(sum_F))
        ax01.plot(Fs[Fs_x_plot_masks, 0], sum_F,
                  '-o',
                  markersize=markersize,
                  color='gray',
                  label='Error')

    if eq_Fs is not None:
        if isinstance(eq_Fs, list):
            # QCP_labels = get_legend_labels('QCR', suffix=legend_labels, length=len(Fs))
            # QEE_labels = get_legend_labels('QER', suffix=legend_labels, length=len(Fs))
            for i, (F, eq_F) in enumerate(zip(Fs, eq_Fs)):
                if add_risk is not None:
                    rect_pareto, rect_total = get_rect_risk(F, add_risk, color_ix=i, external_F=eq_F)
                    ax10.add_patch(rect_pareto)
                    # ax11.add_patch(rect_total)

                ax10.plot(eq_F[:, 0], eq_F[:, 1],
                          '-o',
                          markersize=markersize,
                          color=sns_colors[i],
                          label=legend_labels[i])

                sum_F = np.sum(eq_F[Fs_x_plot_masks[i], :], axis=1)
                y_min_tot_eq = min(y_min_tot_eq, min(sum_F))
                y_max_tot_eq = max(y_max_tot_eq, max(sum_F))
                # ax11.plot(eq_F[Fs_x_plot_masks[i], 0], sum_F,
                #           '-o',
                #           markersize=markersize,
                #           color=sns_colors[i],
                #           label=legend_labels[i])

        else:
            ax10.plot(eq_Fs[:, 0], eq_Fs[:, 1],
                      'o',
                      markersize=markersize,
                      color=sns_colors[0],
                      label='Pareto front - equal weights')

            sum_F = np.sum(eq_Fs[Fs_x_plot_masks], axis=1)
            y_min_tot_eq = min(y_min_tot_eq, min(sum_F))
            y_max_tot_eq = max(y_max_tot_eq, max(sum_F))
            # ax11.plot(Fs[Fs_x_plot_masks, 0], sum_F,
            #           '-o',
            #           markersize=markersize,
            #           color='gray',
            #           label='Error')

    if original_ixs is not None:
        highlight_point(Fs, eq_Fs, axes, original_ixs, color='black', label='Original', edgecolor=True)
    if selected_ixs is not None:
        highlight_point(Fs, eq_Fs, axes, selected_ixs, color='red', label='Selected')

    ax01.set_title('Total Error', fontweight="bold")
    ax01.set_xlabel('Quantile coverage risk')
    ax01.set_ylabel('Total error')
    if xaxis_limit is not None:
        ax01.set_xlim(0, xaxis_limit)
    ax01.set_ylim(y_min_tot * .9, y_max_tot)
    ax01.legend()

    ax00.set_title('Pareto front', fontweight="bold")
    ax00.set_xlabel(xlabel)
    ax00.set_ylabel(ylabel)
    if xaxis_limit is not None:
        ax00.set_xlim(0, xaxis_limit)
    ax00.legend()

    if eq_Fs is not None:
        # ax11.set_title('Total Error - equal weights', fontweight="bold")
        # ax11.set_xlabel('Quantile coverage risk - equal weights')
        # ax11.set_ylabel('Total error')
        # if xaxis_limit is not None:
        #     ax11.set_xlim(0, xaxis_limit)
        # ax11.set_ylim(y_min_tot_eq * .9, y_max_tot_eq)
        # if yaxis_limit_tot_eq is not None:
        #     ax11.set_ylim(y_min_tot_eq * .9, yaxis_limit_tot_eq)
        #
        # ax11.legend()

        ax10.set_title('Error - equal weights', fontweight="bold")
        ax10.set_xlabel('Quantile coverage risk - equal weights')
        ax10.set_ylabel('Quantile estimation risk - equal weights')
        if xaxis_limit is not None:
            ax10.set_xlim(0, xaxis_limit)

        ax10.legend()

    if plot_title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def plot_2D_moo_dual_results(Fs,
                             save=False,
                             file_path=None,
                             selected_ixs=None,
                             original_ixs=None,
                             original_losses=None,
                             figsize=(15, 15),
                             use_date=False,
                             title='Multi objective optimization',
                             col_titles=None,
                             legend_labels=None,
                             add_risk=None,
                             xaxis_limit=None,
                             markersize=5,
                             plot_title=True):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    # ax00, ax01, ax10, ax11 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    # axes = [ax00, ax01, ax10, ax11]

    x_mask = get_x_mask(Fs, xaxis_limit)
    # x_mask = get_x_mask(quantiles_losses, xaxis_limit)
    # labels = get_legend_labels('Error', suffix=legend_labels_suffix, length=len(Fs))
    for i, F in enumerate(Fs):
        if isinstance(F, list):
            for j, f in enumerate(F):
                plot_2D_trace_moo(ax, i, f, x_mask[i][j], sns_colors[j], sns_colors[j],
                                  markersize=markersize,
                                  label=legend_labels[j])

            for j, f in enumerate(F):
                if add_risk is not None:
                    rect_pareto, rect_total = get_rect_risk(f, add_risk, j)
                    ax[0, i].add_patch(rect_pareto)
                    ax[1, i].add_patch(rect_total)

            for j, f in enumerate(F):
                if original_losses is not None:
                    # ix_x_mask = np.arange(f.shape[0]) == original_ixs[i][j]
                    original_loss = np.array(original_losses[i][j]).reshape((1, -1))
                    plot_2D_trace_moo(ax, i, original_loss, [0], 'black', 'black', marker='*',
                                      markersize=26,
                                      label='Original' if j == 0 else None,
                                      edgecolor=(True, j))

                if original_ixs is not None:
                    ix_x_mask = np.arange(f.shape[0]) == original_ixs[i][j]
                    plot_2D_trace_moo(ax, i, f, ix_x_mask, 'black', 'black', marker='*',
                                      markersize=26,
                                      label='Original' if j == 0 else None,
                                      edgecolor=(True, j))
                if selected_ixs is not None:
                    ix_x_mask = np.arange(f.shape[0]) == selected_ixs[i][j]
                    plot_2D_trace_moo(ax, i, f, ix_x_mask, 'red', 'red',
                                      marker='*',
                                      markersize=24,
                                      label='Selected' if j == 0 else None)
        else:
            plot_2D_trace_moo(ax, i, F, x_mask[i], sns_colors[0], 'gray',
                              markersize=markersize)

            if original_ixs is not None:
                ix_x_mask = np.arange(F.shape[0]) == original_ixs[i]
                plot_2D_trace_moo(ax, i, F, ix_x_mask, 'black', 'black', marker='*',
                                  markersize=24,
                                  label='Original')

            if original_losses is not None:
                original_loss = np.array(original_losses[i]).reshape((1, -1))
                plot_2D_trace_moo(ax, i, original_loss, [0], 'black', 'black', marker='*',
                                  markersize=26,
                                  label='Original')

            if selected_ixs is not None:
                ix_x_mask = np.arange(F.shape[0]) == selected_ixs[i]
                plot_2D_trace_moo(ax, i, F, ix_x_mask, 'red', 'red',
                                  marker='*',
                                  markersize=24,
                                  label='Selected')

            if add_risk is not None:
                rect_pareto, rect_total = get_rect_risk(F, add_risk)
                ax[0, i].add_patch(rect_pareto)
                ax[1, i].add_patch(rect_total)

    if col_titles is None:
        col_titles = ['quantile A', 'quantile B']

    for i, label in enumerate(col_titles):
        ax[0, i].set_title('Pareto front: {}'.format(label), fontweight="bold")
        ax[0, i].set_xlabel('Quantile coverage risk')
        ax[0, i].set_ylabel('Quantile estimation risk')

        ax[1, i].set_title('Total risk: {}'.format(label), fontweight="bold")
        ax[1, i].set_xlabel('Quantile coverage risk')
        ax[1, i].set_ylabel('Risk')

        ax[0, i].legend()
        ax[1, i].legend()

    if add_risk is not None:
        for i in range(2):
            for j in range(2):
                handles, labels = ax[i, j].get_legend_handles_labels()
                ax[i, j].legend(
                    handles=handles + [Patch(facecolor='None', edgecolor='g', label='Tolerance',
                                             linestyle='--')],
                    # labels=labels+['risk tolerance'] if add_risk is not None else []
                )
    if plot_title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def get_ixs_risk(F, add_risk):
    tot_risk = np.sum(F, axis=1)
    min_ix = np.argmin(tot_risk)
    min_risk = min(tot_risk)
    max_risk = min_risk * (1 + add_risk)
    l_ix = np.argmin(np.abs(tot_risk[:min_ix] - max_risk))
    r_ix = np.argmin(np.abs(tot_risk[min_ix:] - max_risk)) + min_ix
    return l_ix, r_ix


def get_rect_risk(F, add_risk, color_ix=None, external_F=None):
    if external_F is None:
        external_F = F

    color = 'g' if color_ix is None else sns_colors[color_ix]

    l_ix, r_ix = get_ixs_risk(F, add_risk)

    tot_risk = np.sum(F, axis=1)
    min_risk = min(tot_risk[l_ix:r_ix])
    max_risk = min_risk * (1 + add_risk)
    vertex = (external_F[l_ix, 0], external_F[r_ix, 1])
    width = external_F[r_ix, 0] - external_F[l_ix, 0]
    height = external_F[l_ix, 1] - external_F[r_ix, 1]
    rect_pareto = Rectangle(vertex, width, height, linewidth=3, edgecolor=color, facecolor='none', linestyle='--')

    # print('\nqcr: {}-{}, {}%'.format(np.round(vertex[0], 2),
    #                                  np.round(vertex[0] + width, 2),
    #                                  np.round(width / vertex[0] * 100, 1)))
    # print('qce: {}-{}, {}%'.format(np.round(vertex[1], 2),
    #                                np.round(vertex[1] + height, 2),
    #                                np.round(height / vertex[1] * 100, 1)))

    min_tot_risk = min(np.sum(external_F, axis=1)[l_ix:r_ix])
    max_tot_risk = max(np.sum(external_F, axis=1)[l_ix:r_ix])
    vertex = (external_F[l_ix, 0], min_tot_risk)
    width = external_F[r_ix, 0] - external_F[l_ix, 0]
    height = max_tot_risk - min_tot_risk
    rect_total = Rectangle(vertex, width, height, linewidth=3, edgecolor=color, facecolor='none', linestyle='--')

    return rect_pareto, rect_total


def plot_2D_trace_moo(ax,
                      col,
                      F,
                      x_mask,
                      color_row0,
                      color_row1,
                      marker='-o',
                      markersize=8,
                      label=None,
                      edgecolor=(False, 0)
                      ):
    ax[0, col].plot(F[x_mask, 0], F[x_mask, 1],
                    marker,
                    markersize=markersize,
                    markeredgecolor=sns_colors[edgecolor[1]] if edgecolor[0] else color_row0,
                    markeredgewidth=2 if edgecolor[0] else 1,
                    color=color_row0,
                    label=label)
    ax[1, col].plot(F[x_mask, 0], np.sum(F[x_mask, :], axis=1),
                    marker,
                    markersize=markersize,
                    markeredgecolor=sns_colors[edgecolor[1]] if edgecolor[0] else color_row1,
                    markeredgewidth=2 if edgecolor[0] else 1,
                    color=color_row1,
                    label=label
                    )


def highlight_point(Fs, eq_Fs, axes, ixs, color, label, edgecolor=False):
    ax00, ax01, ax10 = axes
    if isinstance(Fs, list):
        for i, F in enumerate(Fs):
            ax00.plot(F[ixs[i], 0], F[ixs[i], 1],
                      '*',
                      markersize=26,
                      color=color,
                      markeredgecolor=sns_colors[i] if edgecolor else color,
                      markeredgewidth=2,
                      label=label if i == 0 else None)
            ax01.plot(F[ixs[i], 0], np.sum(F, axis=1)[ixs[i]],
                      '*',
                      markersize=26,
                      color=color,
                      markeredgecolor=sns_colors[i] if edgecolor else color,
                      markeredgewidth=2,
                      label=label if i == 0 else None)
    else:
        ax00.plot(Fs[ixs, 0], Fs[ixs, 1],
                  '*',
                  markersize=24,
                  color=color,
                  label=label)
        ax01.plot(Fs[ixs, 0], np.sum(Fs, axis=1)[ixs],
                  '*',
                  markersize=24,
                  color=color,
                  label=label)

    if eq_Fs is not None:
        if isinstance(eq_Fs, list) and isinstance(Fs, list):
            for i, (F, eq_F) in enumerate(zip(Fs, eq_Fs)):
                # ax11.plot(eq_F[ixs[i], 0], np.sum(eq_F, axis=1)[ixs[i]],
                #           '*',
                #           markersize=26,
                #           color=color,
                #           markeredgecolor=sns_colors[i] if edgecolor else color,
                #           markeredgewidth=2,
                #           label=label if i == 0 else None)
                ax10.plot(eq_F[ixs[i], 0], eq_F[ixs[i], 1],
                          '*',
                          markersize=26,
                          color=color,
                          markeredgecolor=sns_colors[i] if edgecolor else color,
                          markeredgewidth=2,
                          label=label if i == 0 else None)
        else:
            # ax11.plot(eq_Fs[ixs, 0], np.sum(eq_Fs, axis=1)[ixs],
            #           '*',
            #           markersize=24,
            #           color=color,
            #           label=label)
            ax10.plot(eq_Fs[ixs, 0], eq_Fs[ixs, 1],
                      'o',
                      markersize=24,
                      color=color,
                      label=label)


def get_x_mask(F_input, xaxis_limit):
    x_mask = []
    for Fs in F_input:
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

        x_mask.append(Fs_x_plot_masks)
    return x_mask

# def highlight_point(Fs, eq_Fs, ax, ixs, color, label):
#     if isinstance(Fs, list) and isinstance(eq_Fs, list):
#         for i, (F, eq_F) in enumerate(zip(Fs, eq_Fs)):
#             ax00.plot(F[ixs[i], 0], F[ixs[i], 1],
#                           '*',
#                           markersize=24,
#                           color=color,
#                           label=label if i == 0 else None)
#             ax01.plot(F[ixs[i], 0], np.sum(F, axis=1)[ixs[i]],
#                           '*',
#                           markersize=24,
#                           color=color,
#                           label=label if i == 0 else None)
#
#             if eq_Fs is not None:
#                 ax11.plot(F[ixs[i], 0], np.sum(eq_F, axis=1)[ixs[i]],
#                               '*',
#                               markersize=24,
#                               color=color,
#                               label=label if i == 0 else None)
#                 ax10.plot(F[ixs[i], 0], eq_F[ixs[i], 0],
#                               'v',
#                               markersize=8,
#                               color=color,
#                               label=label)
#                 ax10.plot(F[ixs[i], 0], eq_F[ixs[i], 1],
#                               '^',
#                               markersize=8,
#                               color=color,
#                               label=label)
#
#     else:
#         ax00.plot(Fs[ixs, 0], Fs[ixs, 1],
#                       '*',
#                       markersize=24,
#                       color=color,
#                       label=label)
#         ax01.plot(Fs[ixs, 0], np.sum(Fs, axis=1)[ixs],
#                       '*',
#                       markersize=24,
#                       color=color,
#                       label=label)
#
#         if eq_Fs is not None:
#             ax11.plot(Fs[ixs, 0], np.sum(eq_Fs, axis=1)[ixs],
#                           '*',
#                           markersize=24,
#                           color=color,
#                           label=label)
#             ax10.plot(Fs[ixs, 0], eq_Fs[ixs, 0],
#                           'v',
#                           markersize=8,
#                           color=color,
#                           label=label)
#             ax10.plot(Fs[ixs, 0], eq_Fs[ixs, 1],
#                           '^',
#                           markersize=8,
#                           color=color,
#                           label=label)
