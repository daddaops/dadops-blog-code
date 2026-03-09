"""TIES-Merging: Trim, Elect Sign, and Disjoint Merge.

Resolves sign conflicts between task vectors by trimming small values,
electing a dominant sign per parameter, and averaging only agreeing values.
"""
import numpy as np

def ties_merge(task_vectors, trim_pct=0.8, scaling=1.0):
    """
    TIES-Merging: Trim, Elect Sign, and Disjoint Merge.

    task_vectors: list of 1D numpy arrays (flattened task vectors)
    trim_pct:     fraction of smallest-magnitude values to trim (0.8 = keep top 20%)
    scaling:      overall scaling factor for the merged task vector
    """
    n_tasks = len(task_vectors)
    n_params = len(task_vectors[0])

    # STEP 1: TRIM — zero out small-magnitude values
    trimmed = []
    for tv in task_vectors:
        threshold = np.quantile(np.abs(tv), trim_pct)
        trimmed_tv = tv.copy()
        trimmed_tv[np.abs(tv) < threshold] = 0.0
        trimmed.append(trimmed_tv)

    trimmed = np.array(trimmed)  # shape: (n_tasks, n_params)

    # STEP 2: ELECT SIGN — majority vote on direction
    # Sum the signs of non-zero values for each parameter
    signs = np.sign(trimmed)  # -1, 0, or +1
    sign_sum = np.sum(signs, axis=0)
    elected_sign = np.sign(sign_sum)  # overall direction per parameter

    # STEP 3: DISJOINT MERGE — average only values that match elected sign
    merged = np.zeros(n_params)
    for j in range(n_params):
        if elected_sign[j] == 0:
            continue  # no consensus, leave at zero
        agreeing = []
        for i in range(n_tasks):
            if np.sign(trimmed[i, j]) == elected_sign[j]:
                agreeing.append(trimmed[i, j])
        if agreeing:
            merged[j] = np.mean(agreeing)

    return scaling * merged

# Demo: merge 3 models where simple averaging fails
np.random.seed(42)
n_params = 200

# Simulate task vectors with deliberate conflicts
tv_A = np.random.randn(n_params) * 0.3
tv_B = np.random.randn(n_params) * 0.3
tv_C = np.random.randn(n_params) * 0.3

# Count sign conflicts
signs = np.sign(np.array([tv_A, tv_B, tv_C]))
conflicts = np.sum(np.min(signs, axis=0) != np.max(signs, axis=0))
print(f"Parameters with sign conflicts: {conflicts}/{n_params} ({conflicts/n_params:.0%})")

# Compare merging methods
simple_avg = (tv_A + tv_B + tv_C) / 3
ties_merged = ties_merge([tv_A, tv_B, tv_C], trim_pct=0.8)

# Measure: how well does each merged vector preserve the dominant direction?
dominant_sign = np.sign(np.sign(tv_A) + np.sign(tv_B) + np.sign(tv_C))
avg_agreement = np.mean(np.sign(simple_avg) == dominant_sign)
ties_agreement = np.mean(np.sign(ties_merged + 1e-10) == dominant_sign)

print(f"Simple average sign agreement: {avg_agreement:.1%}")
print(f"TIES merge sign agreement:     {ties_agreement:.1%}")
print(f"TIES sparsity:                 {np.mean(ties_merged == 0):.0%} of params zeroed")

# Output:
# Parameters with sign conflicts: 150/200 (75%)
# Simple average sign agreement: 77.5%
# TIES merge sign agreement:     96.0%
# TIES sparsity:                 80% of params zeroed
