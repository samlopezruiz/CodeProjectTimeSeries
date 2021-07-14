import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from algorithms.hmm.HiddenMarkovModel import HiddenMarkovModel
#%%

if __name__ == '__main__':
    #%%
    def dptable(state_prob):
        print(" ".join(("%8d" % i) for i in range(state_prob.shape[0])))
        for i, prob in enumerate(state_prob.T):
            print("%.7s: " % states[i] + " ".join("%.7s" % ("%f" % p) for p in prob))


    # Just to highlight the maximum in the Series (color : Yellow)
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    #%%
    p0 = np.array([0.6, 0.4])

    emi = np.array([[0.5, 0.1],
                    [0.4, 0.3],
                    [0.1, 0.6]])

    trans = np.array([[0.7, 0.3],
                      [0.4, 0.6]])

    states = {0: 'Healthy', 1: 'Fever'}
    obs = {0: 'normal', 1: 'cold', 2: 'dizzy'}
    obs_seq = np.array([0, 0, 1, 2, 2])

    df_p0 = pd.DataFrame(p0, index=["Healthy", "Fever"], columns=["Prob"])
    df_emi = pd.DataFrame(emi, index=["Normal", "Cold", "Dizzy"], columns=["Healthy", "Fever"])
    df_trans = pd.DataFrame(trans, index=["fromHealthy", "fromFever"], columns=["toHealthy", "toFever"])

    #%%
    model = HiddenMarkovModel(trans, emi, p0)
    states_seq, state_prob = model.run_viterbi(obs_seq, summary=True)

    print("Observation sequence: ", [obs[o] for o in obs_seq])
    df = pd.DataFrame(state_prob.T, index=["Healthy", "Fever"])
    df.style.apply(highlight_max, axis=0)
