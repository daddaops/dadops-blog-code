import numpy as np

def flash_attention(Q, K, V, B_r=32, B_c=32):
    """
    Flash Attention (FA-2 loop order): exact attention without
    materializing the N×N attention matrix.

    Q, K, V: (N, d) arrays
    B_r: block size for Q (number of query rows per tile)
    B_c: block size for K/V (number of key rows per tile)
    Returns: O (N, d) — same result as standard attention
    """
    N, d = Q.shape
    O = np.zeros((N, d), dtype=np.float32)     # output accumulator (in HBM)
    l = np.zeros((N, 1), dtype=np.float32)     # running denominator per row
    m = np.full((N, 1), -np.inf, dtype=np.float32)  # running max per row

    # Outer loop: iterate over Q blocks
    for i in range(0, N, B_r):
        i_end = min(i + B_r, N)

        # ── Load Q block into SRAM ──
        Q_block = Q[i:i_end]           # (B_r, d)
        O_block = O[i:i_end]           # (B_r, d) accumulator for this Q block
        l_block = l[i:i_end]           # (B_r, 1)
        m_block = m[i:i_end]           # (B_r, 1)

        # Inner loop: iterate over K/V blocks
        for j in range(0, N, B_c):
            j_end = min(j + B_c, N)

            # ── Load K, V block into SRAM ──
            K_block = K[j:j_end]       # (B_c, d)
            V_block = V[j:j_end]       # (B_c, d)

            # ── Compute attention scores for this tile (stays in SRAM) ──
            S_tile = (Q_block @ K_block.T) / np.sqrt(d)   # (B_r, B_c)

            # ── Online softmax update ──
            # Local max for this tile
            m_tile = S_tile.max(axis=-1, keepdims=True)    # (B_r, 1)

            # New running max
            m_new = np.maximum(m_block, m_tile)            # (B_r, 1)

            # Correction factor: rescale old accumulator to new max
            alpha = np.exp(m_block - m_new)            # (B_r, 1)

            # Exponentiated scores, shifted to the running max
            # Note: exp(S - m_tile) * exp(m_tile - m_new) = exp(S - m_new)
            # so we skip the intermediate step and compute directly:
            P_tile = np.exp(S_tile - m_new)            # (B_r, B_c)

            # Update running sum: rescale old sum + new tile sum
            l_new = alpha * l_block + P_tile.sum(axis=-1, keepdims=True)

            # Update output accumulator: rescale old output + new contribution
            O_block = alpha * O_block + P_tile @ V_block

            # Store updated stats
            m_block = m_new
            l_block = l_new

        # ── Normalize and write final output to HBM ──
        O[i:i_end] = O_block / l_block
        l[i:i_end] = l_block
        m[i:i_end] = m_block

    return O

if __name__ == "__main__":
    np.random.seed(42)
    N, d = 256, 64

    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    # Standard attention
    def naive_attention(Q, K, V):
        S = Q @ K.T / np.sqrt(d)
        P = np.exp(S - S.max(axis=-1, keepdims=True))
        P = P / P.sum(axis=-1, keepdims=True)
        return P @ V

    O_naive = naive_attention(Q, K, V)
    O_flash = flash_attention(Q, K, V, B_r=32, B_c=32)

    max_diff = np.max(np.abs(O_naive - O_flash))
    mean_diff = np.mean(np.abs(O_naive - O_flash))

    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Are they equal (tol=1e-5)? {np.allclose(O_naive, O_flash, atol=1e-5)}")

    # Try different block sizes
    for B_r, B_c in [(16, 16), (32, 64), (64, 32), (128, 128)]:
        O_test = flash_attention(Q, K, V, B_r=B_r, B_c=B_c)
        diff = np.max(np.abs(O_naive - O_test))
        print(f"B_r={B_r:>3}, B_c={B_c:>3}: max diff = {diff:.2e}")
