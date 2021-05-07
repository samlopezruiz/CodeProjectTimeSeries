from timeseries.models.lorenz.multivariate.multistep.configs.cnn import cnn_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.models.lorenz.multivariate.multistep.ensemble.func import ensemble_get_multi_step_mv_funcs

def ensemble_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% CNN-LSTM: 0.9467 minmax  (+/- 0.0023) STEPS=6
    model_name = "D-CNN & CNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = [dcnn_mv_configs(steps=6),
                 cnnlstm_mv_configs(steps=6)]
    func_cfg = ensemble_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %% ENSEMBLE: 0.9534 minmax  (+/- 0.0008) STEPS=3
    model_name = "CONV-LSTM & CNN-LSTM & D-CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = [dcnn_mv_configs(steps=3),
                 cnnlstm_mv_configs(steps=3),
                 convlstm_mv_configs(steps=3)]
    func_cfg = ensemble_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %% ENSEMBLE: 0.9601 minmax  (+/- 0.0002)
    model_name = "CONV-LSTM & CNN-LSTM & D-CNN & CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = [convlstm_mv_configs(steps=1),
                 dcnn_mv_configs(steps=1),
                 cnnlstm_mv_configs(steps=1),
                 cnn_mv_configs(steps=1)]
    func_cfg = ensemble_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    return good_configs[keys[steps]]