import time
from itertools import combinations
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from timeseries.models.utils.config import unpack_in_cfg
from timeseries.models.utils.metrics import get_data_error
from timeseries.models.utils.models import get_suffix
from timeseries.models.utils.results import load_results, get_col_and_rename, concat_sort_results, rename_ensemble
from timeseries.plotly.utils import plotly_params_check, plotly_save
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "browser"
import matplotlib.pyplot as plt


def plotly_time_series(df, title=None, save=False, legend=True, file_path=None, size=(1980, 1080), color_col=None,
                       markers='lines+markers', xaxis_title="time", markersize=5, plot_title=True, label_scale=1,
                       **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params
    n_rows = len(set(rows))
    n_cols = 1  # len(set(cols))
    if not params_ok:
        return

    f = len(features)
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True)

    for i in range(f):
        df_ss = df[features[i]].dropna()
        fig.append_trace(
            go.Bar(
                x=df_ss.index,
                y=df_ss,
                orientation="v",
                visible=True,
                showlegend=legend,
                name=features[i],
                opacity=alphas[i]
            ) if type_plot[i] == 'bar' else
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=features[i],
                mode=markers,
                marker=dict(size=markersize,
                            color=None if color_col is None else df[color_col].values,
                            colorscale="Bluered_r"),
                opacity=alphas[i]
            ),
            row=rows[i] + 1,
            col=cols[i] + 1
        )
        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_ts_regime(df, title=None, save=False, legend=True, file_path=None, size=(1980, 1080), regime_col=None,
                     regimes=None, markers='lines+markers', xaxis_title="time", markersize=5, plot_title=True,
                     label_scale=1,
                     **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params
    n_rows = len(set(rows))
    n_cols = 1  # len(set(cols))
    if not params_ok:
        return

    f = len(features)
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True)

    for i in range(f):
        df_ss = df[features[i]].dropna()
        fig.append_trace(
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=features[i],
                mode=markers,
                marker=dict(size=markersize,
                            color=None if regime_col is None else df[regime_col].values,
                            colorscale="Bluered_r"),
                opacity=alphas[i]
            ),
            row=rows[i] + 1,
            col=cols[i] + 1
        )
        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

    colors = ['lightcyan', 'lightblue', 'dodgerblue', 'darkblue', 'black']
    # colors = ['beige', 'palegoldenrod', 'burlywood', 'orange', 'dodgerblue', 'teal']

    if regimes is not None:
        for r, regime in enumerate(regimes):
            for i in range(0, len(regime), 2):
                fig.add_vrect(
                    x0=regime[i], x1=regime[i + 1],
                    fillcolor=colors[r], opacity=0.5,
                    layer="below", line_width=0,
                ),

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_one_series(s, title=None, save=False, legend=True, file_path=None, size=(1980, 1080),
                      markers='lines+markers', xaxis_title="time", label_scale=1, **kwargs):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(
        go.Scatter(
            x=s.index,
            y=np.array(s),
            orientation="v",
            visible=True,
            showlegend=legend,
            name=s.model_name,
        ),
        row=1,
        col=1
    )
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title, legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_phase_plots(df, title=None, save=False, file_path=None, size=(1980, 1080), label_scale=1, legend=True,
                       **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot = params

    if not params_ok:
        return

    f = len(features)
    fig = make_subplots(rows=1, cols=f)
    comb = combinations([0, 1, 2], 2)
    for i, c in enumerate(comb):
        fig.append_trace(
            go.Scatter(
                x=df[features[c[0]]],
                y=df[features[c[1]]],
                visible=True,
                showlegend=False,
            ),
            row=1,
            col=i + 1
        )
        fig['layout']['xaxis' + str(i + 1)]['title'] = features[c[0]]
        fig['layout']['yaxis' + str(i + 1)]['title'] = features[c[1]]

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, title=title)
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    # plotly(fig)
    fig.show()
    time.sleep(1)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_3d(df, title=None, save=False, file_path=None, size=(1980, 1080), legend=True, label_scale=1, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot = params

    if not params_ok:
        return

    f = len(features)
    if f >= 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=df[features[0]],
                    y=df[features[1]],
                    z=df[features[2]],
                    visible=True,
                    showlegend=False,
                    mode='lines',
                )]
        )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1, y=-1, z=1.25)
        )
        fig.update_layout(template="plotly_white",
                          scene=dict(
                              xaxis_title='x',
                              yaxis_title='y',
                              zaxis_title='z'),
                          title=title, scene_camera=camera
                          )
        fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
        fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

        # plotly(fig)
        fig.show()
        time.sleep(2)

        if file_path is not None and save is True:
            plotly_save(fig, file_path, size)


def plotly_acf_pacf(df_acf, df_pacf, save=False, legend=True, file_path=None, size=(1980, 1080),
                    label_scale=1, title_bool=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Bar(
            x=np.arange(len(df_acf)),
            y=df_acf,
            name='ACF',
            orientation="v",
            showlegend=legend,
        ),
        row=1,
        col=1
    )
    fig['layout']['yaxis' + str(1)]['title'] = "Autocorrelation"
    fig.add_trace(
        go.Bar(
            x=np.arange(len(df_pacf)),
            y=df_pacf,
            name='PACF',
            orientation="v",
            showlegend=legend,
        ),
        row=2,
        col=1
    )
    fig['layout']['xaxis' + str(2)]['title'] = 'Lags'
    fig['layout']['yaxis' + str(2)]['title'] = "Partial Autocorrelation"
    fig.update_layout(xaxis_rangeslider_visible=False, title="ACF and PACF" if title_bool else None,
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_history(history, title=None, save=False, legend=True, file_path=None, size=(1980, 1080),
                 markers='lines+markers', label_scale=1, plot_title=True):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(
        go.Scatter(
            x=list(range(len(history.history['loss']))),
            y=history.history['loss'],
            orientation="v",
            visible=True,
            showlegend=legend,
            name='loss',
            mode=markers,
        ),
        row=1,
        col=1
    )
    fig['layout']['xaxis' + str(1)]['title'] = 'loss'
    fig['layout']['yaxis' + str(1)]['title'] = 'epoch'
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_scores(scores, score_type, title=None, save=False, file_path=None, size=(1980, 1080),
                label_scale=1, plot_title=True):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(
        go.Box(y=scores),
        row=1,
        col=1
    )
    fig['layout']['yaxis' + str(1)]['title'] = score_type
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_multiple_scores(scores, score_type, names, title=None, save=False, file_path=None, size=(1980, 1080),
                         label_scale=1, plot_title=True):
    fig = make_subplots(rows=1, cols=len(scores), shared_xaxes=True)
    for i, s in enumerate(scores):
        fig.append_trace(
            go.Box(y=s,
                   name=names[i]),
            row=1,
            col=1
        )
    fig['layout']['yaxis' + str(1)]['title'] = score_type
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_bar_summary(df, errors, title=None, save=False, file_path=None, size=(1980, 1080), shared_yaxes=False,
                     label_scale=1, plot_title=True, n_cols_adj_range=None, showlegend=False):
    if n_cols_adj_range is None:
        n_cols_adj_range = df.shape[1]
    bars = []
    fig = make_subplots(rows=1, cols=df.shape[1], shared_xaxes=True,
                        shared_yaxes=shared_yaxes, subplot_titles=df.columns)
    for i, col in enumerate(df.columns):
        bars.append(
            px.bar(df, x=df.index, y=col, color=df.index,
                   error_y=errors.iloc[:, i] if i < errors.shape[1] else None))

    for i, bar in enumerate(bars):
        for trace in bar.data:
            fig.add_trace(trace, 1, 1 + i)

    if not shared_yaxes:
        for i, col in enumerate(df.columns[:n_cols_adj_range]):
            p = max((max(df[col]) - min(df[col])) / 10, (max(errors.iloc[:, i]) if i < errors.shape[1] else 0))
            f = max(min(df[col]) - p * 1.1, 0)
            c = max(df[col]) + p * 1.1
            fig.update_yaxes(range=[f, c], row=1, col=1 + i)
    else:
        p = max((df.max().max() - df.min().min()) / 10, errors.max().max())
        f = df.min().min() - p * 1.1
        c = df.max().max() + p * 1.1
        fig.update_yaxes(range=[f, c])

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, showlegend=showlegend, barmode="stack")
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_bar_summary_2rows(df, errors, df2, errors2, title=None, save=False, file_path=None, size=(1980, 1080),
                           shared_yaxes=False, label_scale=1, plot_title=True, n_cols_adj_range=None, showlegend=False):
    if n_cols_adj_range is None:
        n_cols_adj_range = df.shape[1]


    fig = make_subplots(rows=2, cols=df.shape[1], shared_xaxes=True,
                        shared_yaxes=shared_yaxes, subplot_titles=list(df.columns) + list(df2.columns))

    bars = []
    for i, col in enumerate(df.columns):
        bars.append(
            px.bar(df, x=df.index, y=col, color=df.index,
                   error_y=errors.iloc[:, i] if i < errors.shape[1] else None))

    for i, bar in enumerate(bars):
        for trace in bar.data:
            fig.add_trace(trace, 1, 1 + i)

    bars = []
    for i, col in enumerate(df2.columns):
        bars.append(
            px.bar(df2, x=df2.index, y=col, color=df2.index,
                   error_y=errors2.iloc[:, i] if i < errors2.shape[1] else None))

    for i, bar in enumerate(bars):
        for trace in bar.data:
            fig.add_trace(trace, 2, 1 + i)

    if not shared_yaxes:
        for i, col in enumerate(df.columns[:n_cols_adj_range]):
            p = max((max(df[col]) - min(df[col])) / 10, (max(errors.iloc[:, i]) if i < errors.shape[1] else 0))
            f = max(min(df[col]) - p * 1.1, 0)
            c = max(df[col]) + p * 1.1
            fig.update_yaxes(range=[f, c], row=1, col=1 + i)

    else:
        p = max((df.max().max() - df.min().min()) / 10, errors.max().max())
        f = df.min().min() - p * 1.1
        c = df.max().max() + p * 1.1
        fig.update_yaxes(range=[f, c])

    p = max((df2.max().max() - df2.min().min()) / 10, errors.max().max())
    f = df2.min().min() - p * 1.1
    c = df2.max().max() + p * 1.1
    fig.update_yaxes(range=[f, c], row=2)

    # ytitles = ['minmax', 'n_params']
    # for i in range(2):
    #     fig['layout']['yaxis' + str(i * df.shape[1] + 1)]['title'] = ytitles[i]

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, showlegend=showlegend, barmode="stack")
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_dc_clusters(dc_df, labels, n_clusters, plot_title=True, title=None, save=False,
                     file_path=None, size=(1980, 1080), label_scale=1, markersize=5):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for c in range(n_clusters):
        fig.append_trace(
            go.Scatter(
                x=dc_df['t'].dropna()[labels == c],
                y=dc_df['tmv'].dropna()[labels == c],
                visible=True,
                showlegend=True,
                mode='markers',
                name='cluster:' + str(c),
                marker=dict(size=markersize),
            ),
            row=1,
            col=1
        )
    fig['layout']['xaxis1']['title'] = 't'
    fig['layout']['yaxis1']['title'] = 'tmv'

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)

    return fig


def plot_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, data, errors):
    file_name = '_'.join(gs_cfg.keys()) + '_' + get_suffix(input_cfg, model_cfg)
    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'MODEL: ' + str(model_cfg),
                     file_path=[in_cfg['image_folder'], file_name], plot_title=in_cfg['plot_title'],
                     save=in_cfg['save_results'], n_cols_adj_range=1)


def plot_multiple_results(res_cfg, steps, var=None):
    dat, err = [], []
    for step in steps:
        res_cfg['steps'] = step
        in_cfg, input_cfg, names, model_cfgs, summary = load_results(res_cfg)
        summary = rename_ensemble(summary)
        if var is None:
            var = (in_cfg['score_type'], 'score_std')

        d, e = get_data_error(summary, in_cfg['score_type'])
        d, e = get_col_and_rename(d, e, var, res_cfg)
        dat.append(d)
        err.append(e)
    data, errors = concat_sort_results(dat, err)

    models_name = res_cfg['suffix']
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(res_cfg)
    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg), plot_title=plot_title,
                     file_path=[image_folder, models_name], showlegend=False, shared_yaxes=True,
                     save=save_results, n_cols_adj_range=data.shape[1])


def plot_multiple_results_2rows(res_cfg, steps, var2):
    data, errors, input_cfg = load_data_err(res_cfg, steps)
    data2, errors2, input_cfg = load_data_err(res_cfg, steps, var2)

    data2 = data2.loc[data.index, :]
    errors2 = errors2.loc[errors.index, :]

    models_name = res_cfg['suffix']
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(res_cfg)
    plot_bar_summary_2rows(data, errors, data2, errors2, title="SERIES: " + str(input_cfg), plot_title=plot_title,
                           file_path=[image_folder, models_name], showlegend=False, shared_yaxes=True,
                           save=save_results, n_cols_adj_range=data.shape[1])


def load_data_err(res_cfg, steps, var=None):
    dat, err = [], []
    for step in steps:
        res_cfg['steps'] = step
        in_cfg, input_cfg, names, model_cfgs, summary = load_results(res_cfg)
        summary = rename_ensemble(summary)
        if var is None:
            var = (in_cfg['score_type'], 'score_std')

        d, e = get_data_error(summary, in_cfg['score_type'])
        d, e = get_col_and_rename(d, e, var, res_cfg)
        dat.append(d)
        err.append(e)
    data, errors = concat_sort_results(dat, err)
    return data, errors, input_cfg
