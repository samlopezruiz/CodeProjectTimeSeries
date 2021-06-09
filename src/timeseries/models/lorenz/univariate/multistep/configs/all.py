from timeseries.models.lorenz.univariate.multistep.configs.cnn import cnn_uv_configs
from timeseries.models.lorenz.univariate.multistep.configs.cnnlstm import cnnlstm_uv_configs
from timeseries.models.lorenz.univariate.multistep.configs.convlstm import convlstm_uv_configs
from timeseries.models.lorenz.univariate.multistep.configs.dcnn import dcnn_uv_configs
from timeseries.models.lorenz.univariate.multistep.configs.gpregress import gpregress_uv_configs
from timeseries.models.lorenz.univariate.multistep.configs.stroganoff import stroganoff_uv_configs
from timeseries.models.lorenz.univariate.multistep.configs.wavenet import wavenet_uv_configs
from timeseries.models.lorenz.univariate.onestep.configs.arima import arima_uv_configs


def all_uv_configs(steps_out, group=None, arima=False):
    names = ["DCNN", "WAVENET", "CNN", "CNN-LSTM", "ConvLSTM", "STROGANOFF", "GP-REGRESS"]

    model_cfgs = []
    model_cfgs.append(dcnn_uv_configs(steps_out)[2])
    model_cfgs.append(wavenet_uv_configs(steps_out)[2])
    model_cfgs.append(cnn_uv_configs(steps_out)[2])
    model_cfgs.append(cnnlstm_uv_configs(steps_out)[2])
    model_cfgs.append(convlstm_uv_configs(steps_out)[2])
    model_cfgs.append(stroganoff_uv_configs(steps_out)[2])
    model_cfgs.append(gpregress_uv_configs(steps_out)[2])

    func_cfgs = []
    func_cfgs.append(dcnn_uv_configs(steps_out)[3])
    func_cfgs.append(wavenet_uv_configs(steps_out)[3])
    func_cfgs.append(cnn_uv_configs(steps_out)[3])
    func_cfgs.append(cnnlstm_uv_configs(steps_out)[3])
    func_cfgs.append(convlstm_uv_configs(steps_out)[3])
    func_cfgs.append(stroganoff_uv_configs(steps_out)[3])
    func_cfgs.append(gpregress_uv_configs(steps_out)[3])


    if group == 'NN':
        names, model_cfgs, func_cfgs = names[:5], model_cfgs[:5], func_cfgs[:5]
    elif group == 'GP':
        names, model_cfgs, func_cfgs = names[-2:], model_cfgs[-2:], func_cfgs[-2:]

    if arima:
        names.append(arima_uv_configs(steps_out)[0])
        model_cfgs.append(arima_uv_configs(steps_out)[2])
        func_cfgs.append(arima_uv_configs(steps_out)[3])

    return names, model_cfgs, func_cfgs