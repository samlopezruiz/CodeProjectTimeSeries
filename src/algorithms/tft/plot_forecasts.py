from os.path import join

import numpy as np
import pandas as pd
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots

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


def append_scatter_trace(fig, df_ss, feature, opacity, color_ix=None, name=None):
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


if __name__ == "__main__":
    base_path = 'outputs/results/volatility/fixed_complete/'
    suffix = '210923-1534'
    targets = pd.read_csv(join(base_path, 'targets'+suffix+'.csv'), index_col=0)
    p10_forecast = pd.read_csv(join(base_path, 'p10_forecast'+suffix+'.csv'), index_col=0)
    p50_forecast = pd.read_csv(join(base_path, 'p50_forecast'+suffix+'.csv'), index_col=0)
    p90_forecast = pd.read_csv(join(base_path, 'p90_forecast'+suffix+'.csv'), index_col=0)

    forecasts = {'targets': targets,
                 'p10_forecast': p10_forecast,
                 'p50_forecast': p50_forecast,
                 'p90_forecast': p90_forecast}
    identifiers = targets['identifier'].unique()

    # %%
    n_output_steps = 5
    features = ['t+' + str(i) for i in range(n_output_steps)]
    label_scale = 1
    markersize = 3
    opacities = np.array([n_output_steps / (i + 1) for i in range(n_output_steps)]) / n_output_steps

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

    # %%
    id = identifiers[25]

    color_counter = 0
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for key in forecasts_grouped:
        df_plot = forecasts_grouped[key][id]

        for i, feature in enumerate(df_plot.columns):
            if key != 'targets' or (key == 'targets' and i == 0):
                append_scatter_trace(fig, df_plot, feature, opacities[i], color_ix=color_counter,
                                     name=key + ' ' + feature)
        color_counter += 1

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, title=id,
                      legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()
