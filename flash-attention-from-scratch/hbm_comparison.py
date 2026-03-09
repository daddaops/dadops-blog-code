def count_hbm_accesses(N, d, B_r, B_c):
    """Count HBM accesses for naive vs Flash Attention."""
    # Naive attention:
    # Read Q, K (2*N*d), write S (N*N), read S (N*N),
    # write P (N*N), read P + V (N*N + N*d), write O (N*d)
    naive = 4*N*d + 4*N*N

    # Flash Attention:
    # Outer loop has ceil(N/B_r) iterations
    # Inner loop has ceil(N/B_c) iterations
    # Each inner iteration: read K_block (B_c*d) + V_block (B_c*d)
    # Each outer iteration: read Q_block (B_r*d), write O_block (B_r*d)
    n_outer = (N + B_r - 1) // B_r
    n_inner = (N + B_c - 1) // B_c

    # Q blocks: loaded once per outer iteration
    flash_q = n_outer * B_r * d
    # K,V blocks: loaded once per (outer, inner) pair
    flash_kv = n_outer * n_inner * 2 * B_c * d
    # O blocks: written once per outer iteration
    flash_o = n_outer * B_r * d
    flash = flash_q + flash_kv + flash_o

    return naive, flash

if __name__ == "__main__":
    d = 64
    B_r, B_c = 128, 128  # realistic SRAM tile size for modern GPUs
    print(f"{'N':>6}  {'Naive':>14}  {'Flash':>14}  {'Ratio':>8}  {'Savings':>8}")
    print("-" * 60)
    for N in [256, 512, 1024, 2048, 4096]:
        naive, flash = count_hbm_accesses(N, d, B_r, B_c)
        ratio = naive / flash
        savings = (1 - flash / naive) * 100
        print(f"{N:>6}  {naive:>14,}  {flash:>14,}  {ratio:>7.1f}x  {savings:>6.1f}%")
