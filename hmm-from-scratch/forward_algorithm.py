import numpy as np
from hmm import casino, forward

true_states, observations = casino.generate(T=30)

log_alpha, log_prob = forward(casino, observations)
print(f"Log P(observations | casino HMM) = {log_prob:.2f}")
print(f"P(observations) = {np.exp(log_prob):.2e}")
# Log P(observations | casino HMM) = -40.17
# P(observations) = 3.57e-18
