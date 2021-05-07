from timeseries.models.lorenz.multivariate.multistep.configs.cnn import cnn_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.ensemble import ensemble_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.gpregress import gpregress_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.stroganoff import stroganoff_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.wavenet import wavenet_mv_configs


def all_mv_configs(steps_out, group=None, ensemble=False):
    names = ["DCNN", "WAVENET", "CNN", "CNN-LSTM", "ConvLSTM", "STROGANOFF", "GP-REGRESS"]

    model_cfgs = []
    model_cfgs.append(dcnn_mv_configs(steps_out)[2])
    model_cfgs.append(wavenet_mv_configs(steps_out)[2])
    model_cfgs.append(cnn_mv_configs(steps_out)[2])
    model_cfgs.append(cnnlstm_mv_configs(steps_out)[2])
    model_cfgs.append(convlstm_mv_configs(steps_out)[2])
    model_cfgs.append(stroganoff_mv_configs(steps_out)[2])
    model_cfgs.append(gpregress_mv_configs(steps_out)[2])

    func_cfgs = []
    func_cfgs.append(dcnn_mv_configs(steps_out)[3])
    func_cfgs.append(wavenet_mv_configs(steps_out)[3])
    func_cfgs.append(cnn_mv_configs(steps_out)[3])
    func_cfgs.append(cnnlstm_mv_configs(steps_out)[3])
    func_cfgs.append(convlstm_mv_configs(steps_out)[3])
    func_cfgs.append(stroganoff_mv_configs(steps_out)[3])
    func_cfgs.append(gpregress_mv_configs(steps_out)[3])


    if group == 'NN':
        names, model_cfgs, func_cfgs = names[:5], model_cfgs[:5], func_cfgs[:5]
    elif group == 'GP':
        names, model_cfgs, func_cfgs = names[-2:], model_cfgs[-2:], func_cfgs[-2:]

    if ensemble:
        names.append(ensemble_mv_configs(steps_out)[0])
        model_cfgs.append(ensemble_mv_configs(steps_out)[2])
        func_cfgs.append(ensemble_mv_configs(steps_out)[3])

    return names, model_cfgs, func_cfgs