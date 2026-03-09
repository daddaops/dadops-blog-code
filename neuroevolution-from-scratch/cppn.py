"""CPPN (Compositional Pattern-Producing Network).

Indirect encoding for HyperNEAT: a small network takes
source/target neuron coordinates and outputs connection weights.
"""
import numpy as np


def cppn_query(x1, y1, x2, y2, params):
    """CPPN: spatial coordinates -> connection weight."""
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)        # distance
    h1 = np.sin(params[0] * d + params[1])            # repetition
    h2 = np.exp(-params[2] * ((x1 + x2) / 2)**2)     # symmetry
    h3 = params[3] * (y2 - y1)                        # gradient
    return np.tanh(params[4]*h1 + params[5]*h2 + params[6]*h3 + params[7])

def generate_weight_matrix(cppn_params, src_pos, tgt_pos):
    """Query CPPN for every source-target pair."""
    W = np.zeros((len(tgt_pos), len(src_pos)))
    for i, (x1, y1) in enumerate(src_pos):
        for j, (x2, y2) in enumerate(tgt_pos):
            W[j, i] = cppn_query(x1, y1, x2, y2, cppn_params)
    return W


if __name__ == "__main__":
    # Substrate: 4x1 input grid, 3x1 output grid
    inputs  = [(-1.5, -1), (-0.5, -1), (0.5, -1), (1.5, -1)]
    outputs = [(-1.0,  1), ( 0.0,  1), (1.0,  1)]

    # These 8 CPPN params would normally be evolved by NEAT
    params = np.array([3.0, 0.5, 2.0, 0.1, 1.2, -0.8, 0.5, 0.0])
    W = generate_weight_matrix(params, inputs, outputs)

    print("Generated 3x4 weight matrix from 8 CPPN params:")
    print(np.round(W, 3))
