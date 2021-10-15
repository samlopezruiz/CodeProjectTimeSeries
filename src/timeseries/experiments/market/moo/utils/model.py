import numpy as np


def get_last_layer_weights(model, layer_name='quantiles'):
    relevant_layers = [l for l in model.model.layers if layer_name in l.name]
    if len(relevant_layers) > 1:
        raise Exception('More than one layer found')
    else:
        last_layer = relevant_layers[0]
        return last_layer.get_weights(), last_layer


def params_conversion_weights(weights):
    shapes = [w.shape for w in weights]
    flatten_dim = [np.multiply(*s) if len(s) > 1 else s[0] for s in shapes]

    ind = np.concatenate([w.flatten() for w in weights])
    params = {
        'shapes': shapes,
        'flatten_dim': flatten_dim
    }
    return ind, params


def reconstruct_weights(ind, params):
    shapes, flatten_dim = params['shapes'], params['flatten_dim']
    reconstruct = []
    for i in range(len(shapes)):
        if i == 0:
            reconstruct.append(ind[:flatten_dim[i]].reshape(shapes[i]))
        else:
            reconstruct.append(ind[flatten_dim[i - 1]:flatten_dim[i - 1] + flatten_dim[i]].reshape(shapes[i]))

    return reconstruct
