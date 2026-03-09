import numpy as np

def basic_composition(epsilon, k):
    """Total privacy cost under basic composition: linear."""
    return k * epsilon

def advanced_composition(epsilon, k, delta=1e-5):
    """Tighter bound: grows as sqrt(k) instead of k.
    From Dwork, Rothblum, Vadhan (2010)."""
    return epsilon * np.sqrt(2 * k * np.log(1.0 / delta))

epsilon_per_query = 0.1

# Compare at k = 100 and k = 1000 queries
for k in [100, 1000]:
    basic = basic_composition(epsilon_per_query, k)
    advanced = advanced_composition(epsilon_per_query, k)
    print(f"After {k} queries at epsilon={epsilon_per_query} each:")
    print(f"  Basic composition:    epsilon_total = {basic:.2f}")
    print(f"  Advanced composition: epsilon_total = {advanced:.2f}")
    print(f"  Improvement: {basic / advanced:.1f}x tighter\n")

# Subsampling amplification
q = 0.01  # use 1% of data per query
effective_eps = 2 * q * epsilon_per_query
print(f"With {q:.0%} subsampling:")
print(f"  Effective epsilon per query: {effective_eps:.4f}")
print(f"  100 queries (basic): {100 * effective_eps:.3f} total")
