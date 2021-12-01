import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from timeseries.data.market.files.utils import load_market
from timeseries.data.market.files.volume import get_full_vol_profile
from timeseries.experiments.market.volume.plotly_vol import plotly_overlap, plot_min_max_vp
from timeseries.experiments.market.volume.process import get_max_min, vol_vp_filter, price_vp_filtering
from timeseries.plotly.volume import plotly_vol_profile, plotly_years_vol_profile


if __name__ == '__main__':
    # %% CONSTANTS
    data_cfg = {'inst': "ES", 'suffix': "2012_1-2021_6_vol", 'sampling': 'minute', 'market': 'cme',
                'src_folder': "vol"}

    df = load_market(data_cfg)


    # %% PLOT YEARLY VOLUME PROFILES
    years = list(range(2013, 2023))
    plotly_years_vol_profile(df, data_cfg['inst'], years)


    # %% PLOT VOLUME PROFILES
    date_input = '2014'
    last_vp = get_full_vol_profile(df, date_input)
    plotly_vol_profile(last_vp)

    # %% LOG AND SMOOTH
    vp_log = np.log(last_vp)
    vp_log_hat = savgol_filter(vp_log, 51, 4)  # window size 51, polynomial order 3
    vp_log_hat = pd.Series(data=vp_log_hat, name=vp_log.name, index=vp_log.index)

    plotly_overlap([vp_log, vp_log_hat])
    plotly_vol_profile(vp_log_hat)

    # %% GET MAX AND MIN
    vp_log_hat_min_max = get_max_min(vp_log_hat)
    plot_min_max_vp(vp_log_hat_min_max, vp_log_hat)

    # %% VOL FILTERING
    vp_log_hat_min_max_filtered = vol_vp_filter(vp_log_hat_min_max, thold=.05)
    plot_min_max_vp(vp_log_hat_min_max_filtered, vp_log_hat)

    # %% PRICE FILTERING
    vp_log_hat_min_max_filtered = price_vp_filtering(vp_log_hat_min_max, thold=3)
    plot_min_max_vp(vp_log_hat_min_max_filtered, vp_log_hat)
