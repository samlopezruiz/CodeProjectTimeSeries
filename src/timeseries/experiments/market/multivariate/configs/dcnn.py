from timeseries.experiments.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
from timeseries.experiments.market.multivariate.architectures.dcnn import dcnn_func


def dcnn_configs(cfg_name):
    funcs = dcnn_func()
    cfgs = {}
    model_cfg = {'name': 'D-CNN', 'verbose': 1, 'use_regimes': False,
                 "n_steps_out": 3, "n_steps_in": 5, 'n_layers': 2, "n_kernel": 2, 'reg': None,
                 "n_filters": 32, 'hidden_channels': 5, "n_batch": 16, "n_epochs": 20}
    cfgs['lorenz_3'] = model_cfg

    model_cfg = {'name': 'D-CNN', 'verbose': 1, 'use_regimes': False,
                 "n_steps_out": 6, "n_steps_in": 9, 'n_layers': 2, "n_kernel": 4, 'reg': None,
                 "n_filters": 64, 'hidden_channels': 5, "n_batch": 32, "n_epochs": 25}
    cfgs['lorenz_6'] = model_cfg

    model_cfg = {'name': 'D-CNN', 'verbose': 1, 'use_regimes': False,
                 "n_steps_out": 1, "n_steps_in": 5, 'n_layers': 2, "n_kernel": 2, 'reg': None,
                 "n_filters": 32, 'hidden_channels': 5, "n_batch": 256, "n_epochs": 25}
    cfgs['lorenz_1'] = model_cfg

    return funcs, cfgs[cfg_name]