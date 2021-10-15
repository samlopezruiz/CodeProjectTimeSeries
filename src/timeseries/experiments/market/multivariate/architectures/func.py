def n_features_3(X):
    return X.shape[3]


def n_features_2(X):
    return X.shape[2]


# steps out are included because lookback consider
# the minimum length needed to have 1 input-output training pair
def lookback_seq_steps(cfg):
    n_seq, n_steps, n_steps_out = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out']
    return n_seq * n_steps + n_steps_out


def lookback_in_steps(cfg):
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    return n_steps_in + n_steps_out


def arg_in_out(cfg, reg_prob):
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    return n_steps_in, n_steps_out, reg_prob
