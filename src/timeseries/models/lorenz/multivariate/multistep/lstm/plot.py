import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.multivariate.multistep.lstm.func import lstm_get_multi_step_mv_funcs
from timeseries.models.utils.models import plot_tf_model


if __name__ == '__main__':
    func_cfg = lstm_get_multi_step_mv_funcs()
    name = "LSTM"
    model_cfg = {"n_steps_in": 6, "n_steps_out": 6, "n_nodes": 50, "n_epochs": 100, "n_batch": 100}
    model = func_cfg[2](model_cfg, n_features=15)
    model.summary()
    plot_tf_model(model, folder='model_img', image_name=name, show_shapes=True)

