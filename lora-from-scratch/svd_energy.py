import numpy as np

# Simulate a "weight update" matrix
# Real weight updates tend to be low-rank because fine-tuning
# nudges weights along a few principal directions
np.random.seed(42)
d = 256  # dimension of the weight matrix

# Create a low-rank-ish update: a few strong directions + small noise
# This mimics what happens during real fine-tuning
strong_directions = 8
U_low = np.random.randn(d, strong_directions) * 2.0
V_low = np.random.randn(strong_directions, d) * 2.0
noise = np.random.randn(d, d) * 0.05
delta_W = U_low @ V_low + noise

# SVD decomposition
U, singular_values, Vt = np.linalg.svd(delta_W)

# How much energy does each rank capture?
total_energy = np.sum(singular_values ** 2)
cumulative_energy = np.cumsum(singular_values ** 2) / total_energy

print("Rank | Cumulative Energy | Parameters")
print("-----|-------------------|----------")
for r in [1, 2, 4, 8, 16, 32]:
    params_full = d * d              # 65,536
    params_lora = r * (d + d)        # much less
    pct = cumulative_energy[r - 1] * 100
    print(f"  {r:>2} | {pct:>16.1f}% | {params_lora:>5} / {params_full} ({params_lora/params_full*100:.1f}%)")
