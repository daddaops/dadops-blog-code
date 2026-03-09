import numpy as np

def kl_divergence(p, q):
    """D_KL(p || q) in nats."""
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    q = np.clip(q, 1e-12, 1.0)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# --- Asymmetry demonstration ---
p = [0.4, 0.3, 0.2, 0.1]  # true distribution
q = [0.25, 0.25, 0.25, 0.25]  # uniform model

print(f"D_KL(p || q) = {kl_divergence(p, q):.4f} nats (forward)")
print(f"D_KL(q || p) = {kl_divergence(q, p):.4f} nats (reverse)")
print(f"They differ! KL is NOT symmetric.\n")

# --- Forward vs reverse KL: fitting to bimodal ---
# Bimodal target: mixture of two Gaussians
x = np.linspace(-6, 6, 1000)
dx = x[1] - x[0]
p_bimodal = 0.5 * np.exp(-0.5*(x+2)**2) + 0.5 * np.exp(-0.5*(x-2)**2)
p_bimodal /= p_bimodal.sum() * dx  # normalize

# Candidate Gaussians: centered between modes vs on one mode
q_cover = np.exp(-0.5 * (x/2.5)**2)  # wide, centered at 0
q_cover /= q_cover.sum() * dx
q_seek = np.exp(-0.5 * (x-2)**2)     # narrow, centered on right mode
q_seek /= q_seek.sum() * dx

p_d, qc_d, qs_d = p_bimodal * dx, q_cover * dx, q_seek * dx

fwd_cover = kl_divergence(p_d, qc_d)
fwd_seek  = kl_divergence(p_d, qs_d)
rev_cover = kl_divergence(qc_d, p_d)
rev_seek  = kl_divergence(qs_d, p_d)

print("Forward KL D_KL(p||q) — mode-covering wins:")
print(f"  q_cover: {fwd_cover:.4f}  q_seek: {fwd_seek:.4f}")
print("Reverse KL D_KL(q||p) — mode-seeking wins:")
print(f"  q_cover: {rev_cover:.4f}  q_seek: {rev_seek:.4f}")
