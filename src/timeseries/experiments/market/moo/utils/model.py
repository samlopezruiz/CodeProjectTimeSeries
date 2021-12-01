import copy

import numpy as np
import pandas as pd

from algorithms.tft2.harness.train_test import compute_moo_q_loss
from algorithms.tft2.utils.nn import dense_layer_output
import tensorflow as tf

def get_last_layer_weights(model, layer_name='quantiles'):
    relevant_layers = [l for l in model.model.layers if layer_name in l.name]
    if len(relevant_layers) > 1:
        raise Exception('More than one layer found')
    else:
        last_layer = relevant_layers[0]
        return last_layer.get_weights(), last_layer

def get_new_weights(original_weights, selected_weights):
    new_weights = copy.deepcopy(original_weights)
    new_weights[0][:, 0] = selected_weights['lq'][:-1]
    new_weights[0][:, 2] = selected_weights['uq'][:-1]
    new_weights[1][0] = selected_weights['lq'][-1]
    new_weights[1][2] = selected_weights['uq'][-1]
    return new_weights

def params_conversion_weights(weights):
    shapes = [w.shape for w in weights]
    flatten_dim = [np.multiply(*s) if len(s) > 1 else s[0] for s in shapes]

    ind = np.concatenate([w.flatten() for w in weights]).reshape(1, -1)
    params = {
        'shapes': shapes,
        'flatten_dim': flatten_dim
    }
    return ind, params


def get_ix_ind_from_weights(weights, ix_weight):
    w = weights[0][:, ix_weight].reshape(1, -1)
    b = weights[1][ix_weight].reshape(1, 1)
    ind = np.hstack([w, b])
    return ind


def reconstruct_weights(ind, params):
    shapes, flatten_dim = params['shapes'], params['flatten_dim']
    reconstruct = []
    ind = ind.reshape(-1, )
    for i in range(len(shapes)):
        if i == 0:
            reconstruct.append(ind[:flatten_dim[i]].reshape(shapes[i]))
        else:
            reconstruct.append(ind[flatten_dim[i - 1]:flatten_dim[i - 1] + flatten_dim[i]].reshape(shapes[i]))

    return reconstruct


def create_output_map(prediction,
                      quantiles,
                      output_size,
                      data_map,
                      time_steps,
                      num_encoder_steps):
    # Extract predictions for each quantile into different entries
    process_map = {
        'p{}'.format(int(q * 100)):
            prediction[Ellipsis, i * output_size:(i + 1) * output_size]
        for i, q in enumerate(quantiles)
    }

    process_map['targets'] = data_map['outputs']

    return {k: format_outputs(process_map[k],
                              data_map,
                              time_steps,
                              num_encoder_steps) for k in process_map}


def format_outputs(prediction,
                   data_map,
                   time_steps,
                   num_encoder_steps):
    """Returns formatted dataframes for prediction."""
    time = data_map['time']
    identifier = data_map['identifier']

    flat_prediction = pd.DataFrame(
        prediction[:, :, 0],
        columns=[
            't+{}'.format(i + 1)
            for i in range(time_steps - num_encoder_steps)
        ])
    cols = list(flat_prediction.columns)
    flat_prediction['forecast_time'] = time[:, num_encoder_steps - 1, 0]
    flat_prediction['identifier'] = identifier[:, 0, 0]

    # Arrange in order
    return flat_prediction[['forecast_time', 'identifier'] + cols]


def run_moo_nn(x,
               quantiles,
               output_size,
               data_map,
               time_steps,
               num_encoder_steps,
               transformer_output,
               w_params,
               loss_to_obj,
               p50_w,
               p50_b,
               output_eq_loss=False):
    new_weights = reconstruct_weights(x, w_params)

    if p50_w is not None and p50_b is not None:
        new_weights[0] = np.vstack([new_weights[0][:, 0],
                                    p50_w,
                                    new_weights[0][:, 1]]).T
        new_weights[1] = np.array([new_weights[1][0],
                                   p50_b,
                                   new_weights[1][1]])

    prediction = dense_layer_output(new_weights, transformer_output)
    unscaled_output_map = create_output_map(prediction,
                                            quantiles,
                                            output_size,
                                            data_map,
                                            time_steps,
                                            num_encoder_steps)

    losses = compute_moo_q_loss(quantiles, unscaled_output_map, output_eq_loss=output_eq_loss)

    if output_eq_loss:
        return loss_to_obj(losses[0]), loss_to_obj(losses[1])
    else:
        return loss_to_obj(losses)


def run_single_w_nn(x,
                    quantiles,
                    output_size,
                    data_map,
                    time_steps,
                    num_encoder_steps,
                    transformer_output,
                    ix_weight,
                    original_weights,
                    overwrite_q=None):

    new_weights = copy.deepcopy(original_weights)
    weights, b = x[:-1], x[-1]
    new_weights[0][:, ix_weight] = weights
    new_weights[1][ix_weight] = b

    prediction = dense_layer_output(new_weights, transformer_output)
    unscaled_output_map = create_output_map(prediction,
                                            quantiles,
                                            output_size,
                                            data_map,
                                            time_steps,
                                            num_encoder_steps)

    losses = compute_moo_q_loss(quantiles,
                                unscaled_output_map,
                                overwrite_q=overwrite_q)

    # if output_eq_loss:
    #     return losses[0][ix_weight, :], losses[1][ix_weight, :]
    # else:
    return losses[ix_weight, :]
