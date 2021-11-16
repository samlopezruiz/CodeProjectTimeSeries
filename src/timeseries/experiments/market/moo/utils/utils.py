import numpy as np

def loss_wo_middle_row(loss):
    return np.array([loss[0, 0], loss[0, 1], loss[2, 0], loss[2, 1]])

def get_loss_to_obj_function(type_func):
    if type_func == 'mean_across_quantiles':
        return lambda x: np.mean(x, axis=0)
    elif type_func == 'ind_loss_woP50':
        return loss_wo_middle_row