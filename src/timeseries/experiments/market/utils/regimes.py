import numpy as np


def append_state(df, n_states, use_regimes):
    if use_regimes:
        cols = ['regime '+str(i) for i in range(n_states)]
        df['state'] = np.argmax(df.loc[:, cols].to_numpy(), axis=1)