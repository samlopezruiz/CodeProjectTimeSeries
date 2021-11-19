import numpy as np
from pymcdm import methods as mcdm_methods
from pymcdm import weights as mcdm_weights
from pymcdm.helpers import rankdata

def loss_wo_middle_row(loss):
    return np.array([loss[0, 0], loss[0, 1], loss[2, 0], loss[2, 1]])

def get_loss_to_obj_function(type_func):
    if type_func == 'mean_across_quantiles':
        return lambda x: np.mean(x, axis=0)
    elif type_func == 'ind_loss_woP50':
        return loss_wo_middle_row
    
def aggregate_qcd_qee(ql):
    ql_2k = np.empty((ql.shape[0], 2))
    ql_2k[:, 0] = ql[:, 0] + ql[:, 2]
    ql_2k[:, 1] = ql[:, 1] + ql[:, 3]
    return ql_2k


def rank_solutions(matrix, weights=None, types=None):
    if weights is None:
        weights = mcdm_weights.equal_weights(matrix)

    if types is None:
        types = np.array([-1] * matrix.shape[1])

    topsis = mcdm_methods.TOPSIS()
    ranks = rankdata(topsis(matrix, np.array(weights), types), reverse=True)

    return np.argsort(ranks)

def sort_1st_col(X, F, eq_F=None):
    ix = np.argsort(F, axis=0)
    X_sorted = X[ix[:, 0], :]
    F_sorted = F[ix[:, 0], :]
    if eq_F is not None:
        eq_F_sorted = eq_F[ix[:, 0], :]
        return X_sorted, F_sorted, eq_F_sorted
    else:
        return X_sorted, F_sorted


def get_selected_ix(quantiles_loss, risk, upper=True):

    valid_keys = ['qcru', 'qeru'] if upper else ['qcrl', 'qerl']
    valid_ix = {}

    for key, value in risk.items():
        if key in valid_keys:
            valid_ix[key] = [np.argmax(np.array(valid_keys) == key), value]

    if not valid_keys:
        raise ValueError('{} does have valid options, must contain {}'.format(risk, valid_keys))

    # consider only first element for bound
    for key, value in valid_ix.items():
        return np.argmin(np.abs(quantiles_loss[:, value[0]] - value[1]))