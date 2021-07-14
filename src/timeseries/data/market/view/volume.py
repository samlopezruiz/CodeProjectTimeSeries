from timeseries.data.market.files.utils import load_market
from timeseries.data.market.files.volume import get_full_vol_profile
from timeseries.plotly.volume import plotly_vol_profile, plotly_years_vol_profile

if __name__ == '__main__':
    data_cfg = {'inst': "ES", 'suffix': "2012_1-2021_6_vol", 'sampling': 'minute', 'market': 'cme',
                'src_folder': "vol"}

    df, vp_prices = load_market(data_cfg)

#%%
    date_input = '2015-01-01 00:00:00'
    date_input = '2022'
    # last_vp = get_full_vol_profile(df, date_input)
    # plotly_vol_profile(last_vp, data_cfg['inst'])

    #%% PLOT YEARLY VOLUME PROFILES
    years = list(range(2013, 2023))
    plotly_years_vol_profile(df, data_cfg['inst'], years)
