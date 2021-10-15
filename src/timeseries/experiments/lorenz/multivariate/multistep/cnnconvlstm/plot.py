import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.configs.cnnconvlstm import cnnconvlstm_mv_configs
from timeseries.experiments.utils.tf import plot_tf_model

if __name__ == '__main__':
    name, input_cfg, model_cfg, func_cfg = cnnconvlstm_mv_configs(steps=3)
    model = func_cfg[2](model_cfg, n_features=3)
    plot_tf_model(model, folder='model_img', image_name=name, show_shapes=True)
