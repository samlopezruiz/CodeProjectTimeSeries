from timeseries.models.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.models.utils.models import plot_tf_model

if __name__ == '__main__':
    name, input_cfg, model_cfg, func_cfg = convlstm_mv_configs(steps=3)
    model = func_cfg[2](model_cfg, n_features=3)
    plot_tf_model(model, folder='model_img', image_name=name, show_shapes=True)
