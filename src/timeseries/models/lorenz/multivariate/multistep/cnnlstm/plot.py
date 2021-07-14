from timeseries.data.lorenz.lorenz import multivariate_lorenz
from timeseries.models.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.models.utils.tf import plot_tf_model

if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=5, positive_offset=False)
    name, input_cfg, model_cfg, func_cfg = cnnlstm_mv_configs(steps=3)
    model, _, _ = func_cfg[1](train, model_cfg)
    plot_tf_model(model, folder='model_img', image_name=name, show_shapes=True)

