import numpy as np

def spiking_forward(spike_input, weights, n_steps, beta=0.9,
                    v_thresh=1.0):
    """Two-layer spiking network: input spikes -> hidden -> output."""
    w1, w2 = weights
    n_hidden = w1.shape[1]
    n_output = w2.shape[1]

    v_hid = np.zeros(n_hidden)
    v_out = np.zeros(n_output)
    out_spikes = np.zeros(n_output)

    for t in range(n_steps):
        # Layer 1: input spikes drive hidden neurons
        i_hid = spike_input[t] @ w1
        v_hid = beta * v_hid + i_hid
        s_hid = (v_hid >= v_thresh).astype(float)
        v_hid = v_hid * (1 - s_hid)  # reset

        # Layer 2: hidden spikes drive output neurons
        i_out = s_hid @ w2
        v_out = beta * v_out + i_out
        s_out = (v_out >= v_thresh).astype(float)
        v_out = v_out * (1 - s_out)
        out_spikes += s_out

    return out_spikes  # spike counts per output neuron

if __name__ == "__main__":
    # 4-input, 8-hidden, 3-output network
    rng = np.random.default_rng(42)
    w1 = rng.normal(0, 0.5, (4, 8))
    w2 = rng.normal(0, 0.5, (8, 3))

    # Rate-encode some inputs
    inputs = np.array([0.8, 0.2, 0.6, 0.4])
    spikes_in = (rng.random((50, 4)) < inputs).astype(float)

    counts = spiking_forward(spikes_in, [w1, w2], n_steps=50)
    print(f"Output spike counts: {counts.astype(int)}")
    print(f"Predicted class: {np.argmax(counts)}")
