import time
from itertools import combinations

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from timeseries.plotly.utils import plotly_params_check, plotly_save


def plotly_time_series(df, title=None, save=False, legend=True, file_path=None, size=(1980, 1080),
                       markers='lines+markers', xaxis_title="time", **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot = params
    n_rows = len(set(rows))
    n_cols = 1 #len(set(cols))
    if not params_ok:
        return

    f = len(features)
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True)

    for i in range(f):
        fig.append_trace(
            go.Bar(
                x=df.index,
                y=df[features[i]],
                orientation="v",
                visible=True,
                showlegend=legend,
                name=features[i],
            ) if type_plot[i] == 'bar' else
            go.Scatter(
                x=df.index,
                y=df[features[i]],
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
                      title=title)
    # plotly(fig)
    fig.show()
    time.sleep(1)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_phase_plots(df, title=None, save=False, file_path=None, size=(1980, 1080), legend=True, **kwargs):
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
    # plotly(fig)
    fig.show()
    time.sleep(1)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_3d(df, title=None, save=False, file_path=None, size=(1980, 1080), legend=True, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, size = params

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
        fig.update_layout(template="plotly_white",
                          scene=dict(
                              xaxis_title='x',
                              yaxis_title='y',
                              zaxis_title='z'),
                          title=title
                          )

        # plotly(fig)
        fig.show()
        time.sleep(2)

        if file_path is not None and save is True:
            plotly_save(fig, file_path, size)


def plotly_acf_pacf(df_acf, df_pacf, save=False, legend=True, file_path=None, size=(1980, 1080)):
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
    fig.update_layout(xaxis_rangeslider_visible=False, title="ACF and PACF")
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig