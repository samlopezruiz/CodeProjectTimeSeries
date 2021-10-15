from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.functions import walk_forward_step_forecast, grid_search
from timeseries.experiments.lorenz.univariate.onestep.mlp.func import mlp_one_step_uv_fit, mlp_one_step_uv_predict, \
    mlp_configs
from timeseries.experiments.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)

    # %% FIT MODEL
    name = "MLP"
    # model configs
    cfg_list = mlp_configs()
    # grid search
    # scores = grid_search(train, test, cfg_list, mlp_one_step_uv_predict, mlp_one_step_uv_fit)
    # print('done')
    # # list top 3 configs
    # for cfg, error in scores[:3]:
    #     print(cfg, error)


