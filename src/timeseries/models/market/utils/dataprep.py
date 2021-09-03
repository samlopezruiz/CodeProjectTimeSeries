import pandas as pd
from numpy import array
import numpy as np

def df_to_np(var):
    return var.to_numpy() if isinstance(var, pd.DataFrame) else var

def to_np(vars):
    np_vars = [df_to_np(var) for var in vars]
    return tuple(np_vars)

def reshape_xy_prob(model_func, model_cfg, x_prob_subsets):
    X, Y, probs = [], [], []
    regs = False
    for i, (x, prob) in enumerate(x_prob_subsets):
        # print(i)
        x, prob = df_to_np(x), df_to_np(prob)
        x_reshaped, y_reshaped, X_reg = model_func['xy_from_train'](x, *model_func['xy_args'](model_cfg, prob))
        X.append(x_reshaped)
        Y.append(y_reshaped)
        if X_reg is not None:
            regs = True
        probs.append(X_reg)

    stacked_x = np.vstack(X)
    stacked_y = np.vstack(Y)
    stacked_prob = np.vstack(probs) if regs else None

    return stacked_x, stacked_y, stacked_prob


def split_mv_seq_multi_step(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix >= len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def step_feature_multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out, n_seq, reg_prob=None):
    # dataset = np.vstack(seqs).transpose()
    if n_steps_in % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_mv_seq_multi_step(seqs, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], n_seq, int(n_steps_in / n_seq), n_features))
    if reg_prob is None:
        return X, y, None
    else:
        assert seqs.shape[0] == reg_prob.shape[0]
        # add dummy y_col
        reg_prob_train_in = np.hstack([reg_prob, np.zeros((reg_prob.shape[0], 1))])
        X_reg, _ = split_mv_seq_multi_step(reg_prob_train_in, n_steps_in, n_steps_out)
        # take last regime to predict next steps
        X_reg_new = [x[-1, :] for x in X_reg]
        X_reg_new = np.vstack(X_reg_new)
        return X, y, X_reg_new


def feature_multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out, reg_prob=None):
    # dataset = np.vstack(seqs).transpose()
    X, y = split_mv_seq_multi_step(seqs, n_steps_in=n_steps_in, n_steps_out=n_steps_out)
    if reg_prob is None:
        return X, y, None
    else:
        assert seqs.shape[0] == reg_prob.shape[0]
        # add dummy y_col
        reg_prob_train_in = np.hstack([reg_prob, np.zeros((reg_prob.shape[0], 1))])
        X_reg, _ = split_mv_seq_multi_step(reg_prob_train_in, n_steps_in, n_steps_out)
        # take last regime to predict next steps
        X_reg_new = [x[-1, :] for x in X_reg]
        X_reg_new = np.vstack(X_reg_new)
        return X, y, X_reg_new
