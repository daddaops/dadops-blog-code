import numpy as np

def stdp_update(pre_times, post_times, w, a_plus=0.005,
                a_minus=0.0055, tau_plus=20.0, tau_minus=20.0,
                w_max=1.0):
    """Apply STDP rule for all spike pairs."""
    for t_pre in pre_times:
        for t_post in post_times:
            dt = t_post - t_pre
            if dt > 0:  # pre before post: potentiate
                dw = a_plus * np.exp(-dt / tau_plus)
                w = min(w + dw, w_max)
            elif dt < 0:  # post before pre: depress
                dw = -a_minus * np.exp(dt / tau_minus)
                w = max(w + dw, 0.0)
    return w

if __name__ == "__main__":
    # Causal pairing: pre fires 5ms before post, repeated 20 times
    pre_spikes = np.arange(0, 400, 20).tolist()    # every 20ms
    post_spikes = np.arange(5, 405, 20).tolist()   # 5ms later each time

    w = 0.3  # initial weight
    weights_over_time = [w]
    for i in range(len(pre_spikes)):
        w = stdp_update([pre_spikes[i]], [post_spikes[i]], w)
        weights_over_time.append(w)

    print(f"Weight: {weights_over_time[0]:.3f} -> {weights_over_time[-1]:.3f}")

    # Anti-causal: post fires 5ms BEFORE pre
    w_anti = 0.3
    for i in range(len(pre_spikes)):
        w_anti = stdp_update([post_spikes[i]], [pre_spikes[i]], w_anti)

    print(f"Anti-causal: {0.3:.3f} -> {w_anti:.3f}")
