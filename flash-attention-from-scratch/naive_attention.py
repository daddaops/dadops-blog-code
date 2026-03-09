import numpy as np

def naive_attention_counted(Q, K, V):
    """Standard attention with HBM access counting."""
    N, d = Q.shape
    hbm_reads = 0
    hbm_writes = 0

    # Step 1: Compute S = Q @ K^T
    # Read Q (N*d) and K (N*d) from HBM
    hbm_reads += 2 * N * d
    S = Q @ K.T / np.sqrt(d)
    # Write S (N*N) to HBM
    hbm_writes += N * N

    # Step 2: Softmax
    # Read S (N*N) from HBM
    hbm_reads += N * N
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)
    # Write P (N*N) to HBM
    hbm_writes += N * N

    # Step 3: Output = P @ V
    # Read P (N*N) and V (N*d) from HBM
    hbm_reads += N * N + N * d
    O = P @ V
    # Write O (N*d) to HBM
    hbm_writes += N * d

    total_accesses = hbm_reads + hbm_writes
    return O, {
        'hbm_reads': hbm_reads,
        'hbm_writes': hbm_writes,
        'total': total_accesses,
        'n_squared_terms': 4 * N * N,   # S write + S read + P write + P read
        'n_d_terms': 4 * N * d          # Q, K reads + V read + O write
    }

if __name__ == "__main__":
    # Example: N=2048, d=64
    np.random.seed(42)
    N, d = 2048, 64
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    O_naive, stats = naive_attention_counted(Q, K, V)
    print(f"Sequence length: {N}, Head dim: {d}")
    print(f"HBM reads:  {stats['hbm_reads']:>12,}")
    print(f"HBM writes: {stats['hbm_writes']:>12,}")
    print(f"Total HBM:  {stats['total']:>12,}")
    print(f"N² terms:   {stats['n_squared_terms']:>12,}  ({100*stats['n_squared_terms']/stats['total']:.0f}%)")
    print(f"Nd terms:   {stats['n_d_terms']:>12,}  ({100*stats['n_d_terms']/stats['total']:.0f}%)")
