import time

import matplotlib.ticker as tick
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from timeseries.data.market.files.volume import get_full_vol_profile


def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val / 1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val / 1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]

    return new_tick_format


def vp_subplot(ax, last_vp, inst, font_size=40, yticks_space=20):
    date = last_vp.name.strftime("%m/%d/%Y")
    vol = np.array(last_vp)[::-1]
    volp = np.array(last_vp.index)[::-1]
    st = time.time()
    b = sns.barplot(x=vol, y=volp, orient='h', ax=ax, color="navy")
    print("Plot created in {} s".format(round(time.time() - st, 2)))

    yticks = ax.yaxis.get_major_ticks()
    for i in range(len(yticks)):
        yticks[i].set_visible(False)
        if float(volp[i]) % yticks_space == 0:
            yticks[i].set_visible(True)

    b.axes.set_title(inst + " Volume Profile " + date, fontsize=font_size * 1.2)
    b.set_xlabel("Volume", fontsize=font_size)
    b.set_ylabel("Price", fontsize=font_size)
    b.tick_params(labelsize=font_size / 1.5)
    ax.set_yticklabels(['{:.0f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));


def plot_full_vol_profile(last_vp, inst, font_size=40, size=(15, 30), yticks_space=20, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    vp_subplot(ax, last_vp, inst, font_size=font_size, yticks_space=yticks_space)
    plt.show()
    return ax


def plot_years_vol_profile(df, years, inst, size=(15, 30), font_size=40, yticks_space=20):
    size = (size[0] * len(years), size[1])
    fig, axes = plt.subplots(1, len(years), figsize=size)
    for i, year in enumerate(years):
        last_vp = get_full_vol_profile(df, str(year))
        vp_subplot(axes[i], last_vp, inst, font_size=font_size, yticks_space=yticks_space)
    plt.show()