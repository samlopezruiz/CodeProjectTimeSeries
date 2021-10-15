from numpy import array
import numpy as np


def split_uv_seq_one_step(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def split_mv_seq_one_step(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix >= len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# split a univariate sequence into samples
def split_uv_seq_multi_step(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


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


def one_step_xy_from_mv(seqs, n_steps=3):
    # dataset = np.vstack(seqs).transpose()
    X, y = split_mv_seq_one_step(seqs, n_steps)
    # flatten input
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    return X, y


def multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out):
    # dataset = np.vstack(seqs).transpose()
    X, y = split_mv_seq_multi_step(seqs, n_steps_in=n_steps_in, n_steps_out=n_steps_out)
    # flatten input
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    return X, y


def feature_one_step_xy_from_uv(dataset, n_steps=3):
    X, y = split_uv_seq_one_step(dataset, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X, y


def feature_one_step_xy_from_mv(seqs, n_steps_in):
    # dataset = np.vstack(seqs).transpose()
    X, y = split_mv_seq_one_step(seqs, n_steps_in)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    return X, y


def feature_multi_step_xy_from_uv(dataset, n_steps_in, n_steps_out):
    X, y = split_uv_seq_multi_step(dataset, n_steps_in=n_steps_in, n_steps_out=n_steps_out)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X, y


def feature_multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out):
    # dataset = np.vstack(seqs).transpose()
    X, y = split_mv_seq_multi_step(seqs, n_steps_in=n_steps_in, n_steps_out=n_steps_out)
    return X, y


def step_feature_one_step_xy_from_uv(seq, n_steps, n_features, n_seq):
    if n_steps % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_uv_seq_one_step(seq, n_steps)
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], n_seq, int(n_steps/n_seq), n_features))
    return X, y


def step_feature_multi_step_xy_from_uv(seq, n_steps_in, n_steps_out, n_features, n_seq):
    if n_steps_in % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_uv_seq_multi_step(seq, n_steps_in, n_steps_out)
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], n_seq, int(n_steps_in/n_seq), n_features))
    return X, y


def step_feature_one_step_xy_from_mv(seqs, n_steps, n_seq):
    # dataset = np.vstack(seqs).transpose()
    if n_steps % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_mv_seq_one_step(seqs, n_steps)
    n_features = X.shape[2]
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], n_seq, int(n_steps/n_seq), n_features))
    return X, y


def step_feature_multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out, n_seq, reg_prob=None):
    # dataset = np.vstack(seqs).transpose()
    if n_steps_in % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_mv_seq_multi_step(seqs, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], n_seq, int(n_steps_in/n_seq), n_features))
    if reg_prob is None:
        return X, y
    else:
        assert seqs.shape[0] == reg_prob.shape[0]
        reg_prob_train_in = np.hstack([reg_prob, np.zeros((reg_prob.shape[0], 1))])
        X_reg, _ = split_mv_seq_multi_step(reg_prob_train_in, 16, 6)
        # take last regime to predict next steps
        X_reg_new = [x[-1, :] for x in X_reg]
        X_reg_new = np.vstack(X_reg_new)
        return X, y, X_reg_new


def row_col_one_step_xy_from_uv(seq, n_steps, n_features, n_seq):
    if n_steps % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_uv_seq_one_step(seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
    X = X.reshape((X.shape[0], n_seq, 1, int(n_steps/n_seq), n_features))
    return X, y


def row_col_multi_step_xy_from_uv(seq, n_steps_in, n_steps_out, n_features, n_seq):
    if n_steps_in % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_uv_seq_multi_step(seq, n_steps_in, n_steps_out)
    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
    X = X.reshape((X.shape[0], n_seq, 1, int(n_steps_in/n_seq), n_features))
    return X, y


def row_col_one_step_xy_from_mv(seqs, n_steps, n_seq):
    # dataset = np.vstack(seqs).transpose()
    if n_steps % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_mv_seq_one_step(seqs, n_steps)
    n_features = X.shape[2]
    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
    X = X.reshape((X.shape[0], n_seq, 1, int(n_steps/n_seq), n_features))
    return X, y


def row_col_multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out, n_seq):
    # dataset = np.vstack(seqs).transpose()
    if n_steps_in % n_seq != 0:
        print('ERROR: n_steps is not divisible by n_seq')
        return
    # split into samples
    X, y = split_mv_seq_multi_step(seqs, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
    X = X.reshape((X.shape[0], n_seq, 1, int(n_steps_in/n_seq), n_features))
    return X, y