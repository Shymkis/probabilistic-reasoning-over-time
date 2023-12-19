#
# Author: Joe Shymanski
# Email:  joe-shymanski@utulsa.edu
#

import numpy as np
import pandas as pd
import random
from CS5313_Localization_Env import localization_env as le

class DBN():
    def __init__(self, prior, transition_model, sensor_model):
        self.P = prior
        self.T = transition_model
        self.S = sensor_model

    def prior_sample(self, N):
        samp = [random.choices(list(p.keys()), p.values(), k=N) for p in self.P]
        return list(zip(*samp))

    def sample_transition(self, state):
        loc = state[0]
        x = loc[0]
        y = loc[1]
        head = state[1]
        samp = [random.choices(list(t[x][y][head].keys()), t[x][y][head].values())[0] for t in self.T]
        samp[0] = (x + samp[0].value[0], y + samp[0].value[1])
        return tuple(samp)

    def evidence_prob(self, evidence, state):
        loc = state[0]
        x = loc[0]
        y = loc[1]
        return self.S[x][y][evidence]


# Page 915
def particle_filtering(S, e, N, dbn):
    W = np.ones(len(S))
    for i in range(N):
        S[i] = dbn.sample_transition(S[i])  # step 1
        W[i] = dbn.evidence_prob(e, S[i])   # step 2
    if not W.any():
        W = np.ones(len(S))
    W /= W.sum()
    return random.choices(S, W, k=N)        # step 3


def get_location_and_heading_probs(S, dims):
    df = pd.DataFrame(S, columns=["location", "heading"])
    loc_probs_tmp = df["location"].value_counts(normalize=True).to_dict()
    head_probs = df["heading"].value_counts(normalize=True).to_dict()

    # Convert location probs dict to a fully complete map including zeros
    loc_probs = np.zeros(dims)
    for loc, prob in loc_probs_tmp.items():
        x = loc[0]
        y = loc[1]
        loc_probs[x][y] = prob

    # Add zero probs for missing headings
    for head in le.Headings:
        if head not in head_probs:
            head_probs[head] = 0

    return loc_probs, head_probs


if __name__ == "__main__":
    seed = 12345
    random.seed(seed)
    env = le.Environment(
        action_bias=0.1,
        observation_noise=0.1,
        action_noise=0.2,
        dimensions=(9,18),
        seed=seed,
        window_size=[400,800]
    )
    N = 50000
    dbn = DBN(
        [env.location_priors, env.heading_priors],
        [env.location_transitions, env.headings_transitions],
        env.observation_tables
    )

    # Start with prior sample
    S = dbn.prior_sample(N)
    while env.running and env.steps < 100:
        # Predict next location
        loc_probs, head_probs = get_location_and_heading_probs(S, env.dimensions)
        env.update(loc_probs, head_probs)

        # Make a move and observe
        obs = tuple(env.move())

        # Update S with new observation
        S = particle_filtering(S, obs, N, dbn)
