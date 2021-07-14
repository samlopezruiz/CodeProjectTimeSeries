

import numpy as np
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt

from algorithms.hmm.func import fitHMM

np.random.seed(42)

if __name__ == '__main__':
    #%%
    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([[0.8, 0.1, 0.1],
                                [0.1, 0.8, 0.1],
                                [0.1, 0.1, 0.8]])
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    X, Z = model.sample(200)

    #%%
    fig, axes = plt.subplots(3)
    axes[0].plot(X[:, 0])
    axes[1].plot(X[:, 1])
    axes[2].plot(Z)
    plt.show()

    #%%
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=3)

    fig, axes = plt.subplots(3)
    axes[0].plot(X[:, 0])
    axes[1].plot(X[:, 1])
    axes[2].plot(Z)
    axes[2].plot(hidden_states)
    plt.show()

    print(mus)
    print(P)

    #%%
    Q = X
    n_components = 3
    n_iter = 100
    shape = np.array(Q).shape
    obs = np.argmax(shape)
    f = np.argmin(shape)
    Q = np.reshape(Q, [shape[obs], shape[f]])
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=n_components, n_iter=n_iter).fit(Q)  # np.reshape(Q, [len(Q), 1])

    # GaussianHMM or GMMHMM
    # classify each observation as state 0 or 1
    hidden_states = model.predict(Q)  # np.reshape(Q, [len(Q), 1])
    hidden_proba = model.predict_proba(Q)
    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)