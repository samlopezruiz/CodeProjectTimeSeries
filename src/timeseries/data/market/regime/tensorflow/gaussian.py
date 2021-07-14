import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
# tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from matplotlib import pylab as plt
import scipy.stats

# %%

if __name__ == '__main__':
    # %%
    true_means = [40., 3., 20.]
    true_scale = [10., 20., 5.]

    num_states = 3
    initial_state_logits = np.zeros([num_states], dtype=np.float32)  # uniform distribution
    daily_change_prob = 0.05
    transition_probs = daily_change_prob / (num_states - 1) * np.ones([num_states, num_states], dtype=np.float32)
    np.fill_diagonal(transition_probs, 1 - daily_change_prob)


    hmm = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(logits=initial_state_logits),
        transition_distribution=tfd.Categorical(probs=transition_probs),
        observation_distribution=tfd.MultivariateNormalDiag(loc=true_means, scale_diag=true_scale),
        num_steps=300)

    observed_counts = hmm.sample()

    plt.plot(observed_counts)
    plt.show()

    # %%
    num_states = 4

    initial_state_logits = np.zeros([num_states], dtype=np.float32)  # uniform distribution

    daily_change_prob = 0.05
    transition_probs = daily_change_prob / (num_states - 1) * np.ones([num_states, num_states], dtype=np.float32)
    np.fill_diagonal(transition_probs, 1 - daily_change_prob)

    print("Initial state logits:\n{}".format(initial_state_logits))
    print("Transition matrix:\n{}".format(transition_probs))

    # %%
    # Define variable to represent the unknown log rates.
    trainable_log_rates = tf.Variable(np.log(np.mean(observed_counts)) + tf.random.normal([num_states]),
                                      name='log_rates')


    hmm = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(logits=initial_state_logits),
        transition_distribution=tfd.Categorical(probs=transition_probs),
        observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
        num_steps=len(observed_counts))

    mvn  = tfd.MultivariateNormalDiag(
        loc=[1., -1],
        scale_diag=[1, 2.])

    #%%
    rate_prior = tfd.LogNormal(5, 5)

    def log_prob():
        return (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) +
                hmm.log_prob(observed_counts))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function(autograph=False)
    def train_op():
        with tf.GradientTape() as tape:
            neg_log_prob = -log_prob()
        grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
        optimizer.apply_gradients([(grads, trainable_log_rates)])
        return neg_log_prob, tf.math.exp(trainable_log_rates)

    #%%
    for step in range(201):
        loss, rates = [t.numpy() for t in train_op()]
        if step % 20 == 0:
            print("step {}: log prob {} rates {}".format(step, -loss, rates))

    print("Inferred rates: {}".format(rates))
    print("True rates: {}".format(true_rates))

    #%%
    # Runs forward-backward algorithm to compute marginal posteriors.
    posterior_dists = hmm.posterior_marginals(observed_counts)
    posterior_probs = posterior_dists.probs_parameter().numpy()

    #%%
    def plot_state_posterior(ax, state_posterior_probs, title):
        ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | counts)')
        ax.set_ylim(0., 1.1)
        ax.set_ylabel('posterior probability')
        ax2 = ax.twinx()
        ln2 = ax2.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
        ax2.set_title(title)
        ax2.set_xlabel("time")
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=4)
        ax.grid(True, color='white')
        ax2.grid(False)


    fig = plt.figure(figsize=(10, 10))
    plot_state_posterior(fig.add_subplot(2, 2, 1),
                         posterior_probs[:, 0],
                         title="state 0 (rate {:.2f})".format(rates[0]))
    plot_state_posterior(fig.add_subplot(2, 2, 2),
                         posterior_probs[:, 1],
                         title="state 1 (rate {:.2f})".format(rates[1]))
    plot_state_posterior(fig.add_subplot(2, 2, 3),
                         posterior_probs[:, 2],
                         title="state 2 (rate {:.2f})".format(rates[2]))
    plot_state_posterior(fig.add_subplot(2, 2, 4),
                         posterior_probs[:, 3],
                         title="state 3 (rate {:.2f})".format(rates[3]))
    plt.tight_layout()
    plt.show()

#%%
    most_probable_states = np.argmax(posterior_probs, axis=1)
    most_probable_rates = rates[most_probable_states]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(most_probable_rates, c='green', lw=3, label='inferred rate')
    ax.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
    ax.set_ylabel("latent rate")
    ax.set_xlabel("time")
    ax.set_title("Inferred latent rate over time")
    ax.legend(loc=4)
    plt.show()

    #%%
    most_probable_states2 = hmm.posterior_mode(observed_counts)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(most_probable_states2, c='green', lw=3, label='inferred rate')
    # ax.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
    ax.set_ylabel("latent rate")
    ax.set_xlabel("time")
    ax.set_title("Inferred latent rate over time")
    ax.legend(loc=4)
    plt.show()

    #%%
