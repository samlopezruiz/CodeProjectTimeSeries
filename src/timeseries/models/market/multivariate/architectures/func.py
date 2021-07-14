def n_features_3(X):
    return X.shape[3]

def seq_steps(cfg):
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    return n_seq * n_steps