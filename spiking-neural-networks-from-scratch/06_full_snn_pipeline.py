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
        i_hid = spike_input[t] @ w1
        v_hid = beta * v_hid + i_hid
        s_hid = (v_hid >= v_thresh).astype(float)
        v_hid = v_hid * (1 - s_hid)

        i_out = s_hid @ w2
        v_out = beta * v_out + i_out
        s_out = (v_out >= v_thresh).astype(float)
        v_out = v_out * (1 - s_out)
        out_spikes += s_out

    return out_spikes

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

    hid_spikes_all, hid_v_all, out_v_all = [], [], []

    for t in range(n_steps):
        i_hid = x[t] @ w1
        v_hid = beta * v_hid + i_hid
        s_hid = (v_hid >= v_thresh).astype(float)
        hid_spikes_all.append(s_hid)
        hid_v_all.append(v_hid.copy())
        v_hid = v_hid * (1 - s_hid)

        i_out = s_hid @ w2
        v_out = beta * v_out + i_out
        s_out = (v_out >= v_thresh).astype(float)
        out_v_all.append(v_out.copy())
        v_out = v_out * (1 - s_out)
        out_counts += s_out

    loss = 0.5 * np.sum((out_counts - target) ** 2)

    d_counts = out_counts - target
    dw2 = np.zeros_like(w2)
    dw1 = np.zeros_like(w1)
    for t in range(n_steps):
        sg_out = fast_sigmoid_surrogate(out_v_all[t], v_thresh)
        d_out = d_counts * sg_out
        dw2 += np.outer(hid_spikes_all[t], d_out)

        sg_hid = fast_sigmoid_surrogate(hid_v_all[t], v_thresh)
        d_hid = (d_out @ w2.T) * sg_hid
        dw1 += np.outer(x[t], d_hid)

    w1 -= lr * dw1 / n_steps
    w2 -= lr * dw2 / n_steps
    return w1, w2, loss

def full_snn_pipeline(x_train, y_train, x_test, y_test,
                      n_hidden=32, n_steps=30, n_epochs=40,
                      lr=0.005, beta=0.85, seed=42):
    """Complete SNN: encode, spike, learn, decode."""
    rng = np.random.default_rng(seed)
    n_in = x_train.shape[1]
    n_out = int(y_train.max()) + 1

    w1 = rng.normal(0, 0.3, (n_in, n_hidden))
    w2 = rng.normal(0, 0.3, (n_hidden, n_out))

    for epoch in range(n_epochs):
        total_loss = 0
        for i in range(len(x_train)):
            inp = np.clip(x_train[i], 0, 1)
            spikes_in = (rng.random((n_steps, n_in)) < inp).astype(float)

            target = np.zeros(n_out)
            target[int(y_train[i])] = n_steps * 0.8

            w1, w2, loss = train_snn_step(
                spikes_in, target, w1, w2, beta=beta,
                lr=lr, n_steps=n_steps
            )
            total_loss += loss

        if (epoch + 1) % 10 == 0:
            correct = 0
            for i in range(len(x_test)):
                inp = np.clip(x_test[i], 0, 1)
                sp = (rng.random((n_steps, n_in)) < inp).astype(float)
                counts = spiking_forward(sp, [w1, w2], n_steps,
                                         beta=beta)
                if np.argmax(counts) == int(y_test[i]):
                    correct += 1
            acc = correct / len(x_test)
            avg_loss = total_loss / len(x_train)
            print(f"Epoch {epoch+1}: loss={avg_loss:.3f}, "
                  f"acc={acc:.1%}")

    return w1, w2

if __name__ == "__main__":
    # Example: simple 3-class classification
    rng = np.random.default_rng(0)
    x = rng.random((150, 4))
    y = np.array([0]*50 + [1]*50 + [2]*50)
    x[:50] += np.array([0.3, 0, 0, 0])
    x[50:100] += np.array([0, 0.3, 0, 0])
    x[100:] += np.array([0, 0, 0.3, 0])

    idx = rng.permutation(150)
    x, y = x[idx], y[idx]
    w1, w2 = full_snn_pipeline(x[:120], y[:120], x[120:], y[120:])
