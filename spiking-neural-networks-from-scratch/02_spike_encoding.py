import numpy as np

def rate_encode(values, n_steps=50, seed=42):
    """Rate coding: value = spike probability per timestep."""
    rng = np.random.default_rng(seed)
    n_neurons = len(values)
    spikes = np.zeros((n_steps, n_neurons))
    for t in range(n_steps):
        spikes[t] = (rng.random(n_neurons) < values).astype(float)
    return spikes

def temporal_encode(values, n_steps=50):
    """Temporal coding: stronger input fires earlier."""
    n_neurons = len(values)
    spikes = np.zeros((n_steps, n_neurons))
    for i, v in enumerate(values):
        if v > 0.01:
            spike_time = int((1 - v) * (n_steps - 1))
            spikes[spike_time, i] = 1.0
    return spikes

if __name__ == "__main__":
    inputs = np.array([0.2, 0.5, 0.9])
    rate_spikes = rate_encode(inputs, n_steps=50)
    temp_spikes = temporal_encode(inputs, n_steps=50)

    for i, val in enumerate(inputs):
        r_count = rate_spikes[:, i].sum()
        t_time = np.where(temp_spikes[:, i])[0]
        print(f"Input {val:.1f}: rate={int(r_count)} spikes/50 steps, "
              f"temporal=spike at t={t_time[0] if len(t_time) else 'none'}")
