from itertools import combinations

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import time

template = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "none"]


def plotly_params_check(df, **kwargs):
    if not isinstance(df, pd.DataFrame):
        print("ERROR: First parameter is not pd.DataFrame")
        return False, None

    params_ok = True
    file_path = kwargs.pop('file_path', None)
    size = kwargs.get('size') if kwargs.get('size', None) is not None else (1980, 1080)
    features = kwargs.get('features') if kwargs.get('features', None) is not None else df.columns
    f = len(features)
    rows = kwargs.get('rows') if kwargs.get('rows', None) is not None else list(range(f))
    cols = kwargs.get('cols') if kwargs.get('cols', None) is not None else list(range(f))
    type_plot = kwargs.get('type_plot') if kwargs.get('type_plot', None) is not None else ["line" for _ in range(f)]
    title = kwargs.pop('title', None)

    if not isinstance(file_path, list) and file_path is not None:
        print("ERROR: file_path type is not a list")
        params_ok = False

    if not isinstance(size, tuple):
        print("ERROR: size is not tuple")
        params_ok = False
    if len(size) != 2:
        print("ERROR: len(size) != 2")
        params_ok = False

    for feature in features:
        if feature not in df.columns:
            print("ERROR: feature ", feature, "not found")
            params_ok = False

    if len(rows) != f:
        print("ERROR: len(rows) != features")
        params_ok = False

    if len(cols) != f:
        print("ERROR: len(cols) != features")
        params_ok = False

    if len(type_plot) != f:
        print("ERROR: len(type_plot) != features")
        params_ok = False

    return params_ok, (features, rows, cols, type_plot, file_path, size, title)


# features, rows, cols, type_plot, title, file_path, size

def plotly_row_traces(df, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, file_path, size, title = params

    if not params_ok:
        return

    f = len(features)
    fig = make_subplots(rows=f, cols=1, shared_xaxes=True)

    for i in range(f):
        fig.append_trace(
            go.Bar(
                x=df.index,
                y=df[features[i]],
                orientation="v",
                visible=True,
                showlegend=False,
            ) if type_plot[i] == 'bar' else
            go.Scatter(
                x=df.index,
                y=df[features[i]],
                visible=True,
                showlegend=False,
            ),
            row=i + 1,
            col=1
        )

    set_y_labels(f, features, fig)
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, title=title)
    plot(fig)
    time.sleep(1)

    if file_path is not None:
        plotly_save(fig, file_path, size)
    return fig


def plotly_phase_plots(df, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, file_path, size, title = params

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
    plot(fig)
    time.sleep(1)

    if file_path is not None:
        plotly_save(fig, file_path, size)
    return fig


def plotly_3d(df, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, file_path, size, title = params

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

        plot(fig)
        time.sleep(2)

        if file_path is not None:
            plotly_save(fig, file_path, size)


def set_y_labels(f, features, fig):
    for i in range(f):
        fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]


def plotly_save(fig, file_path, size):
    print("saving .html and .png")
    if not os.path.exists(file_path[0]):
        os.makedirs(file_path[0])
    image_path = file_path[:-1].copy() + [file_path[-1] + ".png"]
    html_path = file_path[:-1].copy() + [file_path[-1] + ".html"]
    if size is None:
        size = (1980, 1080)
    fig.write_image(os.path.join(*image_path), width=size[0], height=size[1])
    fig.write_html(os.path.join(*html_path))
