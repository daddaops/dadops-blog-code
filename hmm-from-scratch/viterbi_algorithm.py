import numpy as np
from hmm import casino, viterbi

true_states, observations = casino.generate(T=30)
faces = observations + 1
labels = ["F", "L"]

decoded, log_prob = viterbi(casino, observations)

print("Rolls:   ", " ".join(str(f) for f in faces))
print("Viterbi: ", " ".join(labels[s] for s in decoded))
print("Truth:   ", " ".join(labels[s] for s in true_states))
accuracy = np.mean(decoded == true_states)
print(f"Accuracy: {accuracy:.1%}")
# Rolls:    3 1 6 6 6 2 6 6 5 4 6 1 6 6 6 6 6 3 2 5 1 6 3 4 2 3 5 1 3 6
# Viterbi:  F F L L L L L L L L L F L L L L L F F F F F F F F F F F F L
# Truth:    F F L L L L L L L L L F F L L L L L F F F F F F F F F F F L
# Accuracy: 93.3%
