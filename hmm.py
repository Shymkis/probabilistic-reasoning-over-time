#
# Author: Joe Shymanski
# Email:  joe-shymanski@utulsa.edu
#

import numpy as np
from numpy.linalg import inv


# Exercise 15.15
# A professor wants to know if students are getting enough sleep.
# Each day, the professor observes whether the students sleep in class, and whether they have red eyes.
# The professor has the following domain theory:
# - The prior probability of getting enough sleep, with no observations, is 0.7.
prior = np.array([.3, .7]) # P(~EnoughSleep_0), P( EnoughSleep_0)
# - The probability of getting enough sleep on night t is 0.8 given that the student got enough sleep the previous night, and 0.3 if not.
T = np.array([
    [.7,.3], # P(~EnoughSleep_t|~EnoughSleep_{t-1}), P( EnoughSleep_t|~EnoughSleep_{t-1})
    [.2,.8]  # P(~EnoughSleep_t| EnoughSleep_{t-1}), P( EnoughSleep_t| EnoughSleep_{t-1})
])
# - The probability of having red eyes is 0.2 if the student got enough sleep, and 0.7 if not.
RED_EYES = np.array([
    [
        [.3,0], # P(~RedEyes_t|~EnoughSleep_t)
        [0,.8]  # P(~RedEyes_t| EnoughSleep_t)
    ],[
        [.7,0], # P( RedEyes_t|~EnoughSleep_t)
        [0,.2]  # P( RedEyes_t| EnoughSleep_t)
    ],
])
# - The probability of sleeping in class is 0.1 if the student got enough sleep, and 0.3 if not.
SLEEP_IN_CLASS = np.array([
    [
        [.7,0], # P(~SleepInClass_t|~EnoughSleep_t)
        [0,.9]  # P(~SleepInClass_t| EnoughSleep_t)
    ],[
        [.3,0], # P( SleepInClass_t|~EnoughSleep_t)
        [0,.1]  # P( SleepInClass_t| EnoughSleep_t)
    ]
])
# Formulate this information as a dynamic Bayesian network that the professor could use
# to filter or predict from a sequence of observations.
# Then reformulate it as a hidden Markov model that has only a single observation variable.
# Give the complete probability tables for the model.
sensor_model = np.array([RED_EYES, SLEEP_IN_CLASS])


# Exercise 15.17
# For the DBN specified in Exercise 15.15 and for the evidence values
# e_1=not red eyes, not sleeping in class
# e_2=red eyes, not sleeping in class
# e_3=red eyes, sleeping in class
evidence = np.array([
    [0, 0], # e_0 is not used
    [0, 0], # e_1 = [~RedEyes, ~SleepInClass]
    [1, 0], # e_2 = [ RedEyes, ~SleepInClass]
    [1, 1]  # e_3 = [ RedEyes,  SleepInClass]
])
# perform the following computations:
# 1. State estimation: Compute P(EnoughSleep_t|e_{1:t}) for each of t=1,2,3.
# 2. Smoothing: Compute P(EnoughSleep_t|e_{1:3}) for each of t=1,2,3.
# 3. Compare the filtered and smoothed probabilities for t=1 and t=2.
def normalize(probs: np.ndarray):
    return probs/probs.sum(axis=probs.ndim-1, keepdims=True)

# Allows for multiple observation/evidence variables
def O(e: np.ndarray):
    O = np.eye(T.shape[0])
    for var in range(sensor_model.shape[0]):
        val = e[var]
        O = O @ sensor_model[var][val]
    return O

# Page 875, 886
def forward(f: np.ndarray, e: np.ndarray):
    return O(e) @ T.T @ f

# Page 878, 886
def backward(b: np.ndarray, e: np.ndarray):
    return T @ O(e) @ b

# Page 880
def forward_backward(ev: np.ndarray, prior: np.ndarray):
    t = ev.shape[0]
    v_shape = (t,) + prior.shape
    fv = np.zeros(v_shape)
    b = np.ones_like(prior)
    sv = np.zeros(v_shape)
    
    fv[0] = prior
    for i in range(1, t):
        fv[i] = forward(fv[i - 1], ev[i])
    for i in range(t - 1, 0, -1):
        sv[i] = normalize(fv[i] * b)
        b = backward(b, ev[i])
    return normalize(fv), sv

fv, sv = forward_backward(evidence, prior)
print("State estimation:")
print(fv[1:4])
print("Smoothing:")
print(sv[1:4])
print("Differences:")
print(sv[1:3] - fv[1:3])

true_vals = []
for lag in range(2, 6):
    # Page 889
    t = 1
    f = prior
    B = np.eye(T.shape[0])
    e_list = []
    def fixed_lag_smoothing(e_t: np.ndarray, d: int, tol: float = 1e-6):
        global t, f, B, e_list
        e_list.append(e_t)
        O_t = O(e_t)
        if t > d:
            e_t_minus_d = e_list.pop(0)
            f = forward(f, e_t_minus_d)
            O_t_minus_d = O(e_t_minus_d)
            B = inv(O_t_minus_d) @ inv(T) @ B @ T @ O_t
        else:
            B = B @ T @ O_t
        t += 1
        if t > d + 1:
            return normalize(f * B @ np.ones_like(f))

    evidence = np.array([
        [0, 0], # e_0 is not used
        [0, 0], # e_1  = [~RedEyes, ~SleepInClass]
        [1, 0], # e_2  = [ RedEyes, ~SleepInClass]
        [1, 1], # e_3  = [ RedEyes,  SleepInClass]
        [0, 0], # e_4  = [~RedEyes, ~SleepInClass]
        [1, 0], # e_5  = [ RedEyes, ~SleepInClass]
        [1, 1], # e_6  = [ RedEyes,  SleepInClass]
        [0, 0], # e_7  = [~RedEyes, ~SleepInClass]
        [1, 0], # e_8  = [ RedEyes, ~SleepInClass]
        [1, 1], # e_9  = [ RedEyes,  SleepInClass]
        [0, 0], # e_10 = [~RedEyes, ~SleepInClass]
        [1, 0], # e_11 = [ RedEyes, ~SleepInClass]
        [1, 1], # e_12 = [ RedEyes,  SleepInClass]
        [0, 0], # e_13 = [~RedEyes, ~SleepInClass]
        [1, 0], # e_14 = [ RedEyes, ~SleepInClass]
        [1, 1], # e_15 = [ RedEyes,  SleepInClass]
        [0, 0], # e_16 = [~RedEyes, ~SleepInClass]
        [1, 0], # e_17 = [ RedEyes, ~SleepInClass]
        [1, 1], # e_18 = [ RedEyes,  SleepInClass]
        [0, 0], # e_19 = [~RedEyes, ~SleepInClass]
        [1, 0], # e_20 = [ RedEyes, ~SleepInClass]
        [1, 1], # e_21 = [ RedEyes,  SleepInClass]
        [0, 0], # e_22 = [~RedEyes, ~SleepInClass]
        [1, 0], # e_23 = [ RedEyes, ~SleepInClass]
        [1, 1], # e_24 = [ RedEyes,  SleepInClass]
        [0, 0]  # e_25 = [~RedEyes, ~SleepInClass]
    ])
    lv = np.zeros((evidence.shape[0],) + prior.shape)
    while t < evidence.shape[0]:
        fls = fixed_lag_smoothing(evidence[t], lag)
        if t-1-lag >= 0:
            lv[t-1-lag] = fls
    true_vals.append(lv[1:, -1])
print(np.array(true_vals).T)
