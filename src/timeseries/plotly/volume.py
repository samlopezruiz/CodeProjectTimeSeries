import time
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from timeseries.data.market.files.volume import get_full_vol_profile
import plotly.io as pio
pio.renderers.default = "browser"

def plotly_years_vol_profile(df, inst, years):
    fig = make_subplots(rows=1, cols=len(years), shared_yaxes=True,
                        subplot_titles=years)

    for i, year in enumerate(years):
        last_vp = get_full_vol_profile(df, str(year))
        vol = np.array(last_vp) #[::-1]
        volp = np.array(last_vp.index).astype(float) #[::-1]
        fig.append_trace(
            go.Scatter(
                x=vol,
                y=volp,
                orientation="h",
                visible=True,
                showlegend=False,
                fill='tozeroy',
            ),
            row=1,
            col=i + 1
        )

    fig['layout']['yaxis']['title'] = "Price"
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      title=inst + " Volume Profile", yaxis_range=[min(volp), max(volp)])
    fig.show()
    time.sleep(.5)


def plotly_vol_profile(last_vp, inst):
    date_title = last_vp.name.strftime("%m/%d/%Y")
    vol = np.array(last_vp) #[::-1]
    volp = np.array(last_vp.index).astype(float) #[::-1]
    fig = go.Figure(
        data=[go.Scatter(
            x=vol,
            y=volp,
            orientation='h',
            visible=True,
            showlegend=False,
            fill='tozeroy',
        )]
    )

    fig['layout']['yaxis']['title'] = "Price"
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      title=inst + " Volume Profile " + date_title,
                      yaxis_range=[min(volp), max(volp)])
    fig.show()
    time.sleep(.5)
