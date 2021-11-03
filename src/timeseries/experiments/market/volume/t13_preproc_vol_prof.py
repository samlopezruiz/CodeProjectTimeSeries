import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import time

from timeseries.data.market.files.utils import load_data
from timeseries.data.market.files.volume import get_full_vol_profile
from timeseries.plotly.volume import plotly_vol_profile


def plotly_overlap(dfs, modes=None, fills=None):
    date_title = dfs[0].name.strftime("%m/%d/%Y")
    fig = make_subplots(rows=1, cols=1)
    if modes is None:
        modes = ['lines' for _ in range(len(dfs))]

    if fills is None:
        fills = [None for _ in range(len(dfs))]

    ymax = max(dfs[0].index)
    ymin = min(dfs[0].index)
    for i, last_vp in enumerate(dfs):
        vol = np.array(last_vp)[::-1]
        volp = np.array(last_vp.index)[::-1]
        ymax = max(ymax, max(volp))
        ymin = min(ymin, min(volp))
        fig.append_trace(
            go.Scatter(
                x=vol,
                y=volp,
                orientation="v",
                visible=True,
                showlegend=False,
                opacity=0.9,
                mode=modes[i],
                fill=fills[i],
            ),
            row=1,
            col=1
        )

    fig['layout']['yaxis']['title'] = "Price"
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      title=inst + " Volume Profile " + date_title,
                      yaxis_range=[ymin, ymax])
    plot(fig)
    time.sleep(1)


def get_diff_mask(diff_df, thold=0.05):
    diff_mask = diff_df > thold
    diff_mask[0] = True

    in_mask = False
    for i, diff in enumerate(diff_df):
        if diff < thold and not in_mask:
            in_mask = True
            c = 2
            if i > 0:
                diff_mask[i - 1] = False
        elif diff >= thold and in_mask:
            in_mask = False
            if c % 2 == 1:
                diff_mask[i - 1 - c // 2] = True
        elif in_mask:
            c += 1
    return diff_mask


def max_min_mask(ser):
    pos_switch = []
    for i in range(1, len(ser)):
        if ser[i] > ser[i - 1]:
            pos_switch.append(True)
        else:
            pos_switch.append(False)
    pos_switch.append(not pos_switch[-1])
    min_mask = pd.Series(pos_switch, name=ser.name, index=ser.index)
    if sum(min_mask.astype(int).diff().fillna(0).ne(0).astype(int)) > 1:
        print("Max-Min not alternating")
    return ~min_mask, min_mask


def get_max_min(df):
    vp_diff1 = df.diff().fillna(0)

    diff_mask = np.sign(vp_diff1).diff().fillna(0).ne(0)
    return df[diff_mask].copy()


def plot_min_max_vp(df, df2):
    max_mask, min_mask = max_min_mask(df)
    df_max = df[max_mask].copy()
    df_min = df[min_mask].copy()

    plotly_overlap([df2, df_max, df_min],
                   modes=['lines', 'markers', 'markers'],
                   fills=['tozeroy', None, None])


def vol_vp_filter(df, thold=.05):
    v_diff = np.abs(df.diff().fillna(0))
    v_diff_mask = get_diff_mask(v_diff, thold=thold)
    return vp_log_hat_min_max[v_diff_mask].copy()


def price_vp_filtering(df, thold=3):
    ix_diff = pd.Series(df.index.astype(float)).diff().fillna(0)
    ix_diff.index = df.index
    ix_diff_mask = get_diff_mask(ix_diff, thold=thold)
    return vp_log_hat_min_max[ix_diff_mask].copy()


# %% CONSTANTS
contract = "03-21"
ROOT = "../historical_data/compressed"
sample = 'minute'
inst = "ES"
src_folder = "vol"
src_path = os.path.join(ROOT, sample, inst, src_folder)
filename = "ES 2012-2020 vol.csv"

# %% LOAD DATA
df, _ = load_data(src_path, filename=filename)

# %% PLOT VOLUME PROFILES
date_input = '2014'
last_vp = get_full_vol_profile(df, date_input)
plotly_vol_profile(last_vp, inst)

# %% LOG AND SMOOTH
vp_log = np.log(last_vp)
vp_log_hat = savgol_filter(vp_log, 51, 4)  # window size 51, polynomial order 3
vp_log_hat = pd.Series(data=vp_log_hat, name=vp_log.name, index=vp_log.index)

plotly_overlap([vp_log, vp_log_hat])
plotly_vol_profile(vp_log_hat, inst)

# %% GET MAX AND MIN
vp_log_hat_min_max = get_max_min(vp_log_hat)
plot_min_max_vp(vp_log_hat_min_max, vp_log_hat)

# %% VOL FILTERING
vp_log_hat_min_max_filtered = vol_vp_filter(vp_log_hat_min_max, thold=.05)
plot_min_max_vp(vp_log_hat_min_max_filtered, vp_log_hat)

# %% PRICE FILTERING
vp_log_hat_min_max_filtered = price_vp_filtering(vp_log_hat_min_max, thold=3)
plot_min_max_vp(vp_log_hat_min_max_filtered, vp_log_hat)
