"""
Evaluation metrics shared across every model family. Unlike the data/split
logic, this code was byte-identical (or comment-only differences) across
all scripts checked -- CNN, MLP, RF, GAM, LR, and all eval_*.py -- so this
extraction carries no behavioural ambiguity.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

# np.trapz was deprecated in numpy>=1.24 and removed in numpy>=2.0 (renamed
# to np.trapezoid). The original scripts used np.trapz directly, which
# would crash on any modern numpy install -- this picks whichever the
# installed numpy provides, same trapezoidal integration either way.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def compute_auprgc(y_true, y_score):
    """
    Area under the precision-recall-gain curve (Flach & Kull, 2015-style
    "gain" transform of precision/recall relative to the base rate `pi`).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pi = y_true.mean()
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    precG = np.zeros_like(precision)
    recG = np.zeros_like(recall)
    mask_prec = precision > 0
    mask_rec = recall > 0
    precG[mask_prec] = (precision[mask_prec] - pi) / ((1 - pi) * precision[mask_prec])
    recG[mask_rec] = (recall[mask_rec] - pi) / ((1 - pi) * recall[mask_rec])

    mask = (precG >= 0) & (recG >= 0)
    precG_plot = precG[mask]
    recG_plot = recG[mask]

    sorted_indices = np.argsort(recG_plot)
    recG_sorted = recG_plot[sorted_indices]
    precG_sorted = precG_plot[sorted_indices]

    return _trapz(precG_sorted, recG_sorted)


def compute_metrics(y_true, y_proba):
    """
    The rocauc + prgauc bundle computed at the end of every validation/eval
    script. Convenience wrapper -- not a behaviour change, just avoids
    repeating the same two calls everywhere.
    """
    return {
        "rocauc": roc_auc_score(y_true, y_proba),
        "prgauc": compute_auprgc(y_true, y_proba),
    }
