


import numpy as np
from hmmlearn import hmm

if __name__ == '__main__':
    #%%
    X1 = np.random.random((10, 2))  # 2-D
    X2 = np.random.random((5, 2))   # must have the same ndim as X1.
    X = np.concatenate([X1, X2])
    lengths = np.array([len(X1), len(X2)])
    model = hmm.GaussianHMM(n_components=2).fit(X, lengths)
    hidden_states = model.predict(X, lengths)

    #%%
    X1 = np.random.random((10, 1))  # 2-D
    X2 = np.random.random((10, 1))  # must have the same ndim as X1.
    Y = np.concatenate([X1, X2], axis=1)
    lengthsY = [10]
    # modelY = hmm.GaussianHMM(n_components=2).fit(Y, lengthsY)
    # hidden_states = modelY.predict(Y, lengthsY)

    #%%
    modelY1 = hmm.GaussianHMM(n_components=2).fit(Y)
    hidden_states1 = modelY1.predict(Y)

