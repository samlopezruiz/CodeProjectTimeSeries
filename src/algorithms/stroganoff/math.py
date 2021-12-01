import numpy as np
from sklearn.metrics import mean_squared_error


def covert_to_array(x1, x2):
    if not isinstance(x1, np.ndarray) and not isinstance(x1, list):
        x1, x2 = np.array([x1]).reshape(-1, 1), np.array([x2]).reshape(-1, 1)
    else:
        x1, x2 = np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1)
    return x1, x2


def Xmat_from_inputX(x1, x2):
    # print(x1.shape, x2.shape)
    x1, x2 = covert_to_array(x1, x2)
    xp = np.ones((x1.shape[0], 1))
    xp = np.append(xp, x1, axis=1)  # x1
    xp = np.append(xp, x2, axis=1)  # x2
    xp = np.append(xp, (xp[:, 1] * xp[:, 2]).reshape(-1, 1), axis=1)  # x1 * x2
    xp = np.append(xp, (xp[:, 1] * xp[:, 1]).reshape(-1, 1), axis=1)  # x1 ^ 2
    xp = np.append(xp, (xp[:, 1] * xp[:, 2]).reshape(-1, 1), axis=1)  # x2 ^ 2
    return xp


def derive_a(X, y):
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)


def G(x1, x2, a):
    return np.matmul(Xmat_from_inputX(x1, x2), a)


def score_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
