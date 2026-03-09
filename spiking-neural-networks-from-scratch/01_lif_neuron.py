import numpy as np

def simulate_lif(current, dt=1.0, tau_m=20.0, v_rest=0.0,
                 v_thresh=1.0, v_reset=0.0, r=1.0, t_ref=4):
    """Simulate a Leaky Integrate-and-Fire neuron."""
    n_steps = len(current)
    voltage = np.zeros(n_steps)
    spikes = np.zeros(n_steps)
    v = v_rest
    ref_counter = 0

    for t in range(n_steps):
        if ref_counter > 0:
            ref_counter -= 1
            v = v_reset
        else:
            dv = (-(v - v_rest) + r * current[t]) * dt / tau_m
            v += dv
            if v >= v_thresh:
                spikes[t] = 1.0
                v = v_reset
                ref_counter = t_ref
        voltage[t] = v

    return voltage, spikes

if __name__ == "__main__":
    # Constant current above threshold
    current = np.ones(200) * 1.8
    voltage, spikes = simulate_lif(current)
    spike_times = np.where(spikes)[0]
    print(f"Spike count: {int(spikes.sum())}")
    print(f"Firing rate: {spikes.sum() / len(current) * 1000:.0f} Hz")
    print(f"Spike times: {spike_times[:8]}...")
