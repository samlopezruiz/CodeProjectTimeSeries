import time

import numpy as np
import pandas as pd

from timeseries.data.market.files.utils import load_market
from timeseries.experiments.market.volume.plotly_vol import plotly_vol_profile_levels, plotly_cols_in_df
from timeseries.experiments.market.volume.process import get_vol_profiles_history_levels, get_levels_and_vol_profiles, \
    create_vp_levels_indicators
from timeseries.experiments.utils.files import save_df, save_vars

if __name__ == '__main__':
    # %% CONSTANTS
    vol_data_cfg = {'inst': "ES", 'suffix': "2012_1-2021_6_vol", 'sampling': 'minute', 'market': 'cme',
                    'src_folder': "vol"}
    market_data_cfg = {'inst': "ES", 'suffix': "2012_1-2021_6", 'sampling': 'minute', 'market': 'cme',
                       'src_folder': "data"}

    indicator_cfg = {'price': 'ESc', 'n_levels': 5}

    market = load_market(market_data_cfg)
    vol = load_market(vol_data_cfg)

    # %%
    price_history = market[indicator_cfg['price']]

    print('Filtering min max points in volume profile...')
    t0 = time.time()
    vol_profile_levels, vol_profile_is_max = get_vol_profiles_history_levels(vol,
                                                                             normalize_profile=True,
                                                                             filter_thold=3,
                                                                             windows_size=51,
                                                                             poly_order=4)

    print('Volume level processing time: {}s'.format(round(time.time() - t0, 2)))

    distance_df, inverse_distance_df, norm_vol_df, is_max_df = create_vp_levels_indicators(price_history,
                                                                                           vol_profile_levels,
                                                                                           vol_profile_is_max,
                                                                                           indicator_cfg)

    # %% Save as compressed file

    volume_profile_levels_complete = pd.concat([inverse_distance_df,
                                                norm_vol_df,
                                                is_max_df], axis=1)
    save_vars(volume_profile_levels_complete,
              file_path=['res', 'Vol_{}levels_{}_{}'.format(indicator_cfg['n_levels'],
                                                            indicator_cfg['price'],
                                                            market_data_cfg['suffix'])])

    # %% PLOT
    n_plot = 3000
    test_ix = 200000

    plot_df = inverse_distance_df.iloc[test_ix - n_plot:test_ix, :]

    plotly_cols_in_df(plot_df,
                      float_y_axis=False,
                      swap_xy=True,
                      modes=['markers' for _ in range(len(plot_df.columns))],
                      markersize=3)

    plot_df = distance_df.iloc[test_ix - n_plot:test_ix, :]

    plotly_cols_in_df(plot_df,
                      float_y_axis=False,
                      swap_xy=True,
                      modes=['markers' for _ in range(len(plot_df.columns))],
                      markersize=3)
