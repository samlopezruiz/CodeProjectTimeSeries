from algorithms.hmm.func import fitHMM
from algorithms.hmm.plot import plotTimeSeries, plotDistribution
from timeseries.data.market.files.utils import load_market
from timeseries.plotly.plot import plotly_ts_regime
from timeseries.preprocessing.func import ln_returns


if __name__ == '__main__':
    #%%
    name = "DIRECTIONAL CHANGE REGIMES"
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'regime_img', 'results_folder': 'results'}
    input_cfg = {'preprocess': True}

    data_cfg = {'inst': "ES", 'suffix': "2012-2020", 'sampling': 'minute',
                'src_folder': "data", 'market': 'cme'}

    # %% LOAD DATA
    es, features_es = load_market(data_cfg)

#%%
    data_from = '2019-12-01 08:30:00'
    data_to = '2020-05-30 15:30:00'
    df = es.loc[data_from:data_to].copy()
    df['ES_r'] = ln_returns(df['ESc'])
    df.dropna(inplace=True)

    #%%
    vars = ['ES_r', 'atr']
    X = df.loc[:, vars]
    hidden_states, mus, sigmas, P, logProb, model = fitHMM(X, 100)

    #%%
    # plotTimeSeries(df['ES_r'], hidden_states, 'log(Flow at State Line)', 'StateTseries_Log.png')
    # plotDistribution(df['ES_r'], mus, sigmas, P, 'MixedGaussianFit_Log.png')

    #%%
    df['state'] = hidden_states
    save_folder = in_cfg['image_folder']
    save_plots = in_cfg['save_results']
    plotly_ts_regime(df, features=['ESc', 'atr'], rows=[0, 1], markers='lines+markers', regimes=None, save=save_plots, regime_col='state',
                     file_path=[save_folder, 'hmm'],
                     size=(1980//1.5, 1080//1.5), markersize=2)