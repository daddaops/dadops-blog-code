"""
Parameter Budget Analysis

Shows why SwiGLU uses h = 8d/3 hidden dimension to match classic FFN
parameter count, and verifies against LLaMA 7B's actual dimensions.

Blog post: https://dadops.dev/blog/ffn-from-scratch/
"""
import numpy as np

d_model = 4096  # LLaMA 7B model dimension

# Theoretical hidden dim
h_theory = 8 * d_model / 3
print(f"Theoretical: {h_theory:.1f}")  # 10922.7

# Real LLaMA 7B: round to nearest multiple of 256
h_llama7b = 11008  # closest multiple of 256 to 10922.7
print(f"LLaMA 7B:   {h_llama7b}")

# Parameter comparison
classic_params = 2 * d_model * (4 * d_model)
swiglu_params  = 3 * d_model * h_llama7b
print(f"\nClassic FFN params per layer: {classic_params:,}")
print(f"SwiGLU FFN params per layer:  {swiglu_params:,}")
print(f"Difference: {(swiglu_params/classic_params - 1)*100:.1f}%")
