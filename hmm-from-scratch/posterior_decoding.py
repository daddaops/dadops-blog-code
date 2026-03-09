import numpy as np
from hmm import casino, forward, viterbi, posterior_decode

true_states, observations = casino.generate(T=30)
faces = observations + 1
labels = ["F", "L"]

decoded, _ = viterbi(casino, observations)
gamma, posterior_path = posterior_decode(casino, observations)

print("Rolls:     ", " ".join(str(f) for f in faces))
print("Posterior: ", " ".join(labels[s] for s in posterior_path))
print("Viterbi:   ", " ".join(labels[s] for s in decoded))
print("Truth:     ", " ".join(labels[s] for s in true_states))
print(f"\nP(Loaded) at each step:")
print(" ".join(f"{gamma[t,1]:.2f}" for t in range(len(observations))))
# Posterior and Viterbi agree on most positions
# P(Loaded) is high (>0.5) during loaded stretches, low during fair stretches
