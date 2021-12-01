import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs2
from timeseries.experiments.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.experiments.utils.tf import plot_tf_model

if __name__ == '__main__':
    name, input_cfg, model_cfg, func_cfg = dcnn_mv_configs(steps=3)
    func_cfg = dcnn_get_multi_step_mv_funcs2()
    model_cfg['n_nodes'] = 10
    model = func_cfg[2](model_cfg, n_features=3)
    plot_tf_model(model, folder='model_img', image_name=name+'half', show_shapes=True)
