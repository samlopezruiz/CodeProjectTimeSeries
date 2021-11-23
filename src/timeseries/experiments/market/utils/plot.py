import time

import numpy as np
import pandas as pd
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from timeseries.plotly.utils import plotly_save

pio.renderers.default = "browser"

colors = [
    '#1f77b4',  # muted blue
    '#2ca02c',  # cooked asparagus green
    '#ff7f0e',  # safety orange
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def plot_forecast_intervals(forecasts_grouped,
                            n_output_steps,
                            id,
                            additional_vars=[],
                            additional_rows=[],
                            additional_data=None,
                            markersize=3,
                            fill_max_opacity=0.1,
                            label_scale=1,
                            title='',
                            mode='light',
                            save=False,
                            file_path=None,
                            size=(1980, 1080),
                            save_png=False,
                            x_range=None,
                            y_range=None,
                            ):

    steps = list(forecasts_grouped[list(forecasts_grouped.keys())[0]][id].columns)
    opacities = np.array([n_output_steps / (i + 1) for i in range(n_output_steps)]) / n_output_steps
    fill_opacities = (np.array([n_output_steps / (i + 1) for i in range(n_output_steps)])
                      / n_output_steps) * fill_max_opacity

    prob_forecasts = sorted(forecasts_grouped.keys())
    if 'p50' in prob_forecasts:
        prob_forecasts.remove('p50')
    if 'targets' in prob_forecasts:
        prob_forecasts.remove('targets')

    pairs = [(prob_forecasts[i], prob_forecasts[len(prob_forecasts) - 1 - i])
             for i in range(len(prob_forecasts) // 2)]

    # plot only one pair of bounds
    for p, pair in enumerate(pairs):
        lower_bound = forecasts_grouped[pair[0]][id]
        upper_bound = forecasts_grouped[pair[1]][id]

        if len(additional_rows) > 0 and max(additional_rows) > 0:
            fig = make_subplots(rows=max(additional_rows)+1, cols=1, shared_xaxes=True)
        else:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        if len(additional_vars) > 0 and additional_data is not None:
            ix = forecasts_grouped['targets'][id].index
            add_data = additional_data.loc[ix, :]
            for i, var in enumerate(additional_vars):
                fig.add_trace(
                    go.Scatter(
                        x=add_data.index,
                        y=add_data[var],
                        visible=True,
                        showlegend=True,
                        name=var,
                        mode='lines+markers',
                        opacity=1,
                        line=dict(color=colors[4 + i], width=2),
                        marker=dict(size=markersize,
                                    color=colors[4 + i]),
                    ),
                    row=additional_rows[i] + 1,
                    col=1
                )

        for i, step in enumerate(steps[::-1]):
            if mode == 'light':
                fillcolor = 'rgba(0, 0, 0, {})'.format(fill_opacities[::-1][i])
            else:
                fillcolor = 'rgba(255, 255, 255, {})'.format(fill_opacities[::-1][i])

            for b, bound in enumerate([lower_bound, upper_bound]):
                fig.add_trace(
                    go.Scatter(
                        x=bound.index,
                        y=bound[step],
                        visible=True,
                        showlegend=False,
                        mode='lines',
                        fill='tonexty' if b == 1 else None,
                        fillcolor=fillcolor,
                        line=dict(color=fillcolor),
                        marker=dict(size=markersize, color=fillcolor),
                    ),
                    row=1,
                    col=1
                )
        if 'p50' in forecasts_grouped.keys():
            for i, step in enumerate(steps):
                fig.add_trace(
                    go.Scatter(
                        x=forecasts_grouped['p50'][id][step].index,
                        y=forecasts_grouped['p50'][id][step],
                        visible=True,
                        showlegend=True,
                        name='{} pred in t{}'.format(step[:-3], str(-(i+1))),
                        mode='lines+markers',
                        opacity=opacities[i],
                        line=dict(color=colors[0], width=2),
                        marker=dict(size=markersize,
                                    color=colors[0]),
                    ),
                    row=1,
                    col=1
                )
        if 'targets' in forecasts_grouped.keys():
            fig.add_trace(
                go.Scatter(
                    x=forecasts_grouped['targets'][id][steps[0]].index,
                    y=forecasts_grouped['targets'][id][steps[0]],
                    visible=True,
                    showlegend=True,
                    name='target',
                    mode='lines+markers',
                    opacity=1,
                    line=dict(color=colors[3], width=3),
                    marker=dict(size=markersize,
                                color=colors[3]),
                ),
                row=1,
                col=1
            )



        fig.update_layout(template="plotly_white" if mode == 'light' else 'plotly_dark',
                          xaxis_rangeslider_visible=False, title=title,
                          legend=dict(font=dict(size=18 * label_scale)))

        if y_range is not None:
            fig.update_layout(yaxis=dict(range=y_range))

        if x_range is not None:
            fig.update_layout(xaxis=dict(range=x_range))

        fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
        fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
        fig.show()
        time.sleep(1.5)

        if file_path is not None and save is True:
            plotly_save(fig, file_path, size, save_png=save_png)


def append_scatter_trace(fig, df_ss, feature, opacity, markersize=5, color_ix=None, name=None):
    name = feature if name is None else name
    fig.append_trace(
        go.Scatter(
            x=df_ss.index,
            y=df_ss[feature],
            visible=True,
            showlegend=True,
            name=name,
            opacity=opacity,
            mode='lines+markers',
            line=dict(color=None if color_ix is None else colors[color_ix]),
            marker=dict(size=markersize,
                        color=None if color_ix is None else colors[color_ix]),
        ),
        row=1,
        col=1
    )


def group_forecasts(forecasts, n_output_steps, target_col):
    if target_col:
        features = ['{} t+{}'.format(target_col, i + 1) for i in range(n_output_steps)]
    else:
        features = ['t+{}'.format(i + 1) for i in range(n_output_steps)]

    forecasts_grouped = {}
    for key, df in forecasts.items():

        identifiers_forecasts = {}
        for id, df_grouped in df.groupby('identifier'):
            df_grouped = df_grouped.set_index('forecast_time', inplace=False)
            shifted = {}
            for i, feature in enumerate(features):
                shifted[feature] = df_grouped[feature].shift(i)
            identifiers_forecasts[id] = pd.DataFrame(shifted)

        forecasts_grouped[key] = identifiers_forecasts

    return forecasts_grouped


def plot_individual_forecast(forecasts_grouped, n_output_steps, id=0, label_scale=1, markersize=3, title=''):
    color_counter = 0

    opacities = np.array([n_output_steps / (i + 1) for i in range(n_output_steps)]) / n_output_steps

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for key in forecasts_grouped:
        df_plot = forecasts_grouped[key][id]

        for i, feature in enumerate(df_plot.columns):
            if key != 'targets' or (key == 'targets' and i == 0):
                append_scatter_trace(fig, df_plot, feature, opacities[i], color_ix=color_counter,
                                     name=key + ' ' + feature, markersize=markersize)
        color_counter += 1

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, title=title,
                      legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()