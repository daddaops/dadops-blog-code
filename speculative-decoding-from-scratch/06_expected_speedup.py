"""Expected speedup analysis for speculative decoding across acceptance rates and K values."""


def expected_speedup(alpha, K, c=0.1):
    """
    alpha: acceptance rate (0 to 1)
    K:     speculation length (number of draft tokens)
    c:     draft_latency / target_latency (typically 0.05-0.2)
    """
    if abs(alpha - 1.0) < 1e-10:
        expected_tokens = K + 1
    else:
        expected_tokens = (1 - alpha ** (K + 1)) / (1 - alpha)

    cost = K * c + 1   # K draft calls + 1 target verify
    return expected_tokens / cost


# Sweep parameters
print(f"{'α':>6} {'K=2':>8} {'K=3':>8} {'K=5':>8} {'K=8':>8}")
print("-" * 40)
for alpha in [0.5, 0.7, 0.8, 0.9, 0.95]:
    row = f"{alpha:>6.2f}"
    for K in [2, 3, 5, 8]:
        s = expected_speedup(alpha, K, c=0.1)
        row += f" {s:>7.2f}x"
    print(row)
