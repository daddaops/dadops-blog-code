import numpy as np

def fast_sigmoid_surrogate(v, v_thresh=1.0, k=25.0):
    """Surrogate gradient: smooth approximation of spike derivative."""
    u = v - v_thresh
    return 1.0 / (1.0 + k * np.abs(u)) ** 2

def train_snn_step(x, target, w1, w2, beta=0.9, v_thresh=1.0,
                   lr=0.01, n_steps=25):
    """One training step: forward with spikes, backward with surrogates."""
    n_hid = w1.shape[1]
    n_out = w2.shape[1]
    v_hid, v_out = np.zeros(n_hid), np.zeros(n_out)
    out_counts = np.zeros(n_out)

    # Collect states for backward pass
    hid_spikes_all, hid_v_all, out_v_all = [], [], []

    for t in range(n_steps):
        # Hidden layer
        i_hid = x[t] @ w1
        v_hid = beta * v_hid + i_hid
        s_hid = (v_hid >= v_thresh).astype(float)
        hid_spikes_all.append(s_hid)
        hid_v_all.append(v_hid.copy())
        v_hid = v_hid * (1 - s_hid)  # reset spiked neurons

        # Output layer
        i_out = s_hid @ w2
        v_out = beta * v_out + i_out
        s_out = (v_out >= v_thresh).astype(float)
        out_v_all.append(v_out.copy())
        v_out = v_out * (1 - s_out)
        out_counts += s_out

    # Loss: MSE on spike counts vs target
    loss = 0.5 * np.sum((out_counts - target) ** 2)

    # Backward pass with surrogate gradients
    d_counts = out_counts - target
    dw2 = np.zeros_like(w2)
    dw1 = np.zeros_like(w1)
    for t in range(n_steps):
        # Output surrogate gradient
        sg_out = fast_sigmoid_surrogate(out_v_all[t], v_thresh)
        d_out = d_counts * sg_out
        dw2 += np.outer(hid_spikes_all[t], d_out)

        # Hidden surrogate gradient
        sg_hid = fast_sigmoid_surrogate(hid_v_all[t], v_thresh)
        d_hid = (d_out @ w2.T) * sg_hid
        dw1 += np.outer(x[t], d_hid)

    w1 -= lr * dw1 / n_steps
    w2 -= lr * dw2 / n_steps
    return w1, w2, loss

if __name__ == "__main__":
    # Demo: one training step on random data
    rng = np.random.default_rng(42)
    n_in, n_hid, n_out, n_steps = 4, 8, 3, 25
    w1 = rng.normal(0, 0.3, (n_in, n_hid))
    w2 = rng.normal(0, 0.3, (n_hid, n_out))
    x = (rng.random((n_steps, n_in)) < 0.5).astype(float)
    target = np.array([0.0, 20.0, 0.0])

    w1, w2, loss = train_snn_step(x, target, w1, w2, n_steps=n_steps)
    print(f"Loss after one step: {loss:.3f}")
    print(f"W1 shape: {w1.shape}, W2 shape: {w2.shape}")
