"""Comprehensive evaluation metrics suite.

Implements confusion matrix, precision/recall/F1, Matthews Correlation
Coefficient, and ROC-AUC via the Mann-Whitney U statistic.
"""
import numpy as np

def confusion_counts(y_true, y_pred):
    """Return TP, FP, FN, TN from binary predictions."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return tp, fp, fn, tn

def precision_recall_f1(y_true, y_pred):
    """Precision, recall, and their harmonic mean."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

def matthews_corrcoef(y_true, y_pred):
    """MCC: balanced metric using all four confusion matrix cells."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    num = tp * tn - fp * fn
    den = np.sqrt(float((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)))
    return num / den if den > 0 else 0

def roc_auc(y_true, scores):
    """AUC via the Mann-Whitney U statistic."""
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    count = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n: count += 1
            elif p == n: count += 0.5
    return count / (len(pos_scores) * len(neg_scores))

# Imbalanced fraud detection: 1% positive rate
y_true = np.array([0]*990 + [1]*10)
y_pred = np.zeros(1000, dtype=int)  # always predict "no fraud"
p, r, f1 = precision_recall_f1(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print(f"Accuracy: {np.mean(y_true == y_pred):.1%}")  # 99.0%
print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")  # all 0
print(f"MCC: {mcc:.3f}")  # 0.000 — correctly says model is useless
