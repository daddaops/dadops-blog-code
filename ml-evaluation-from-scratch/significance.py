"""Statistical significance tests for model comparison.

Implements paired t-test, Nadeau & Bengio corrected test, and
McNemar's test, showing how naive tests inflate false positives.
"""
import numpy as np
from scipy import stats

def paired_ttest(scores_a, scores_b):
    """Naive paired t-test on CV fold scores.
    WARNING: underestimates variance due to overlapping training sets."""
    diffs = scores_a - scores_b
    t_stat = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    p_value = 2 * stats.t.sf(abs(t_stat), df=len(diffs) - 1)
    return t_stat, p_value

def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected test.
    Accounts for non-independence of CV folds."""
    k = len(scores_a)
    diffs = scores_a - scores_b
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    # The correction: inflate variance by (1/k + n_test/n_train)
    corrected_var = (1/k + n_test/n_train) * var_d
    t_stat = mean_d / np.sqrt(corrected_var) if corrected_var > 0 else 0
    p_value = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    return t_stat, p_value

def mcnemar_test(y_true, preds_a, preds_b):
    """McNemar's test on per-example disagreements."""
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B right
    if (b + c) == 0:
        return 0.0, 1.0
    chi2 = (b - c) ** 2 / (b + c)
    p_value = stats.chi2.sf(chi2, df=1)
    return chi2, p_value

# The shocking example
scores_a = np.array([0.88, 0.92, 0.89, 0.91, 0.87,
                     0.93, 0.90, 0.86, 0.91, 0.89])
scores_b = np.array([0.93, 0.91, 0.92, 0.91, 0.91,
                     0.91, 0.93, 0.91, 0.92, 0.91])

_, p_naive = paired_ttest(scores_a, scores_b)
_, p_corr  = corrected_resampled_ttest(scores_a, scores_b, 900, 100)

print(f"2% gap — Naive p={p_naive:.3f}, Corrected p={p_corr:.3f}")
# 2% gap — Naive p=0.030, Corrected p=0.110
# Naive says significant, but the corrected test says NO!
