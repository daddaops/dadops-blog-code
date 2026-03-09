import numpy as np

# Simulate a trained LoRA layer
d, k, rank = 512, 512, 8
alpha = 16
W_base = np.random.randn(d, k) * 0.01
A = np.random.randn(rank, k) * 0.1    # trained values
B = np.random.randn(d, rank) * 0.05   # trained values (no longer zeros)

# Method 1: Separate computation (during training)
x = np.random.randn(1, k)
out_separate = x @ W_base.T + (alpha / rank) * (x @ A.T @ B.T)

# Method 2: Merged weight (for deployment)
W_merged = W_base + (alpha / rank) * (B @ A)
out_merged = x @ W_merged.T

# They're identical
print(f"Max difference: {np.max(np.abs(out_separate - out_merged)):.2e}")
print(f"Outputs match: {np.allclose(out_separate, out_merged)}")

# The economics
base_size_gb = 7e9 * 2 / 1e9  # 7B params in fp16
adapter_size_mb = 20e6 * 2 / 1e6  # 20M LoRA params in fp16
print(f"\nBase model:      {base_size_gb:.0f} GB")
print(f"One adapter:     {adapter_size_mb:.0f} MB")
print(f"10 variants:")
print(f"  Full FT:       {base_size_gb * 10:.0f} GB")
print(f"  LoRA adapters: {base_size_gb + adapter_size_mb * 10 / 1000:.1f} GB")
