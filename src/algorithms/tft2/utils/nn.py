import numpy as np


def dense_layer_output(weights, X):
    b = np.repeat(weights[1].reshape(1, -1), X.shape[1], axis=0)
    ans = []
    for x in X:
        ans.append(np.expand_dims(np.matmul(x, weights[0]) + b, 0))

    return np.concatenate(ans, axis=0)
