import time

from timeseries.data.market.files.utils import load_market
from timeseries.experiments.market.volume.plotly_vol import plotly_vol_profile_levels, plotly_cols_in_df
from timeseries.experiments.market.volume.process import get_vol_profiles_history_levels, get_levels_and_vol_profiles

if __name__ == '__main__':
    # %% CONSTANTS
    data_cfg = {'inst': "ES", 'suffix': "2012_1-2021_6_vol", 'sampling': 'minute', 'market': 'cme',
                'src_folder': "vol"}

    df = load_market(data_cfg)

    # %%
    normalize_profile = True
    print('Filtering min max points in volume profile...')
    t0 = time.time()

    vol_profile_levels, vol_profile_is_max = get_vol_profiles_history_levels(df,
                                                                             normalize_profile=True,
                                                                             filter_thold=3,
                                                                             windows_size=51,
                                                                             poly_order=4)

    # %% PLOT HEATMAP
    volume_profile = vol_profile_levels * (vol_profile_is_max.astype(float) * 2 - 1)
    formatted_df = volume_profile.copy()
    formatted_df.columns = formatted_df.columns.strftime('%m/%d/%Y')  # %H:%M')
    plotly_vol_profile_levels(formatted_df,
                              file_path=['img', 'ES_volume_profile_levels'],
                              save_html=False,
                              title='ES Volume Profile Levels')

    # %% COMPARE LEVELS WITH PROFILE
    date_input = '09/14/2015'

    vol_profile_and_levels = get_levels_and_vol_profiles(date_input,
                                                         df,
                                                         vol_profile_levels,
                                                         vol_profile_is_max)
    plotly_cols_in_df(vol_profile_and_levels,
                      modes=['lines', 'markers', 'markers'],
                      fills=None)
