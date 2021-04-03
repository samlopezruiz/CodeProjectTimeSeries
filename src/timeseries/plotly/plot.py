import time
from itertools import combinations
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from timeseries.plotly.utils import plotly_params_check, plotly_save
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib.pyplot as plt


def plotly_time_series(df, title=None, save=False, legend=True, file_path=None, size=(1980, 1080),
                       markers='lines+markers', xaxis_title="time", plot_title=True, label_scale=1, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot = params
    n_rows = len(set(rows))
    n_cols = 1 #len(set(cols))
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
            ) if type_plot[i] == 'bar' else
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=features[i],
                mode=markers,
            ),
            row=rows[i] + 1,
            col=cols[i] + 1
        )
        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18*label_scale)))

    fig.update_xaxes(tickfont=dict(size=14*label_scale), title_font=dict(size=18*label_scale))
    fig.update_yaxes(tickfont=dict(size=14*label_scale), title_font=dict(size=18*label_scale))

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
            name=s.name,
        ),
        row=1,
        col=1
    )
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title, legend=dict(font=dict(size=18*label_scale)))
    fig.update_xaxes(tickfont=dict(size=14*label_scale), title_font=dict(size=18*label_scale))
    fig.update_yaxes(tickfont=dict(size=14*label_scale), title_font=dict(size=18*label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_phase_plots(df, title=None, save=False, file_path=None, size=(1980, 1080), label_scale=1, legend=True, **kwargs):
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
                      legend=dict(font=dict(size=18*label_scale)))
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