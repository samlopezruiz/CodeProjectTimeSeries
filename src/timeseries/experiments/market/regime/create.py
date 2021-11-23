from algorithms.hmm.func import append_hmm_states
from algorithms.hmm.plot import plot_hmm
from timeseries.data.market.files.utils import load_multiple_markets
from timeseries.experiments.market.preprocess.func import add_features
from timeseries.experiments.market.split.func import time_subset
from timeseries.experiments.market.utils.save import save_hmm
from timeseries.plotly.plot import plotly_ts_candles, plotly_time_series

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': True,
              'verbose': 1,
              'plot_title': False,
              'image_folder': 'img',
              'results_folder': 'res'}

    mkt_data_cfg = {'inst': "ES", 'suffix': "2012_5-2021_6", 'sampling': 'day',
                    'src_folder': "data", 'market': 'cme', 'data_from': '2011-12', 'data_to': '2021-12'}
    fed_data_cfg = {'inst': "FED", 'sampling': 'day', 'suffix': "2012_1-2021_6", 'market': 'fed',
                    'src_folder': "data"}
    mkt_data_cfg2 = {'inst': "NQ", 'suffix': "2012_5-2021_6", 'sampling': 'day',
                     'src_folder': "data", 'market': 'cme'}

    # %%
    price_col = mkt_data_cfg['inst'] + 'c'
    resampling = 'D'
    data_cfgs = [mkt_data_cfg, fed_data_cfg, mkt_data_cfg2]
    df = load_multiple_markets(data_cfgs, resampling='D', ffill=True)
    df_ss = time_subset(df, mkt_data_cfg)
    add_features(df_ss, macds=[price_col], returns=[price_col])

    plot_features = [price_col + '_r', 'volume', 'atr', '^VIX', 'DGS10']
    # plotly_ts_candles(df_ss, features=plot_features, instrument=mkt_data_cfg['inst'],
    #                   rows=[i for i in range(len(plot_features))], adjust_height=(True, 0.6))

    # %%
    hmm_cfg = {'n_states': 5,
               'regime_col': 'state',
               'hmm_vars': ['ESc_r', 'T10Y2Y', '^VIX'],
               'label_cfg': ('^VIX', 'min'),
               'resample_plot': False}

    df_reg, n_regimes, df_proba = append_hmm_states(df_ss,
                                                    hmm_cfg['hmm_vars'],
                                                    hmm_cfg['n_states'],
                                                    label_cfg=hmm_cfg['label_cfg'],
                                                    regime_col=hmm_cfg['regime_col'])

    result = {
        'data': df_reg,
        'n_regimes': n_regimes,
        'df_proba': df_proba,
        'hmm_cfg': hmm_cfg,
        'data_cfgs': data_cfgs
    }
    save_hmm(result, in_cfg, hmm_cfg)

    # plotly_time_series(df_reg, features=['ESo'], color_col='state')

    plot_hmm(df_reg,
             price_col,
             hmm_cfg['hmm_vars'],
             hmm_cfg['n_states'],
             in_cfg,
             regime_col=hmm_cfg['regime_col'],
             resample=hmm_cfg['resample_plot'],
             n_regimes=n_regimes,
             label_scale=1
             )
