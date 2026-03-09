"""Architecture Scaling Comparison — Attention vs MegaByte vs SSM costs.

Code Block 5: Shows how SSM (MambaByte) achieves linear scaling.
"""

def compute_costs(N_values, P=8):
    """Compare operation counts: attention, MegaByte, and SSM."""
    results = []
    for N in N_values:
        flat = N * N                          # O(N^2) self-attention
        mega = (N // P) ** 2 + N * P          # MegaByte
        ssm = N * 64                          # O(N*d_state) for SSM scan
        results.append((N, flat, mega, ssm))
    return results

N_values = [256, 512, 1024, 2048, 4096, 8192, 16384]
results = compute_costs(N_values)

print(f"{'N':>6} {'Attention':>12} {'MegaByte':>12} {'SSM':>12}")
print("-" * 46)
for N, flat, mega, ssm in results:
    print(f"{N:>6} {flat:>12,} {mega:>12,} {ssm:>12,}")

# Blog claims at N=8192:
#   Attention: 67,108,864
#   MegaByte:   1,114,112  (60x less)
#   SSM:          524,288  (128x less)
