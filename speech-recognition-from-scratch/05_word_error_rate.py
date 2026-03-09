"""Word Error Rate (WER) computation using dynamic programming."""
import numpy as np


def word_error_rate(reference, hypothesis):
    """Compute WER using dynamic programming (Levenshtein distance at word level).
    reference: string (ground truth transcript)
    hypothesis: string (model output)
    Returns: WER as float, and (substitutions, deletions, insertions)
    """
    ref = reference.split()
    hyp = hypothesis.split()
    N, M = len(ref), len(hyp)

    # DP table: dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = np.zeros((N + 1, M + 1), dtype=int)
    for i in range(N + 1): dp[i, 0] = i  # deletions
    for j in range(M + 1): dp[0, j] = j  # insertions

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j-1],  # substitution
                                    dp[i-1, j],     # deletion
                                    dp[i, j-1])     # insertion

    # Backtrace to count S, D, I
    i, j, S, D, I = N, M, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i,j] == dp[i-1,j-1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i,j] == dp[i-1,j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1

    wer = (S + D + I) / N if N > 0 else 0.0
    return wer, (S, D, I)


if __name__ == "__main__":
    ref = "set a timer for five minutes"
    hyp = "set the timer for five minute"
    wer, (s, d, i) = word_error_rate(ref, hyp)
    print(f"Reference:  '{ref}'")
    print(f"Hypothesis: '{hyp}'")
    print(f"S={s}, D={d}, I={i}, N={len(ref.split())}")
    print(f"WER = ({s}+{d}+{i})/{len(ref.split())} = {wer:.1%}")
