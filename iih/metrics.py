"""Accuracy, macro-F1, hand-off metrics and a paired bootstrap over test ids."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    assert len(y_true) == len(y_pred) > 0
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred):
    # Averaged over the classes present in y_true: cooking_query has no test examples and would
    # otherwise contribute a meaningless 0.
    return float(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0))


def auroc(known_scores, unknown_scores):
    """Higher score = more likely unknown. Unknown is the positive class. 0.5 = chance."""
    y = np.r_[np.zeros(len(known_scores)), np.ones(len(unknown_scores))]
    return float(roc_auc_score(y, np.r_[known_scores, unknown_scores]))


def threshold_at_retention(known_scores, retention=0.95):
    """Score threshold that keeps `retention` of known utterances (kept = score <= threshold)."""
    return float(np.quantile(np.asarray(known_scores), retention))


def acceptance_at_retention(known_scores, unknown_scores, retention=0.95):
    """Share of unknowns let through when the threshold keeps `retention` of knowns (FPR@95%TPR)."""
    t = threshold_at_retention(known_scores, retention)
    return float((np.asarray(unknown_scores) <= t).mean())


def at_threshold(known_scores, unknown_scores, t):
    """(known retention, unknowns caught) at a fixed threshold chosen elsewhere."""
    return float((np.asarray(known_scores) <= t).mean()), float((np.asarray(unknown_scores) > t).mean())


def bootstrap(stat, n, reps=2000, seed=7):
    """95% percentile interval of stat(idx) over resampled index arrays; idx is shared across arms."""
    rng = np.random.default_rng(seed)
    vals = np.sort([stat(rng.integers(0, n, n)) for _ in range(reps)])
    return float(vals[int(0.025 * reps)]), float(vals[int(0.975 * reps) - 1])
