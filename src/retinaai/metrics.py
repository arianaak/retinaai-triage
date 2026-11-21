"""Clinically meaningful metrics for DR grading (numpy/sklearn only)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score


def compute_metrics(y_true, probs, referable_threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = probs.argmax(1)
    ref_true = y_true >= 2
    p_ref = probs[:, 2:].sum(1)
    ref_pred = p_ref >= referable_threshold
    tp = int((ref_pred & ref_true).sum())
    tn = int((~ref_pred & ~ref_true).sum())
    fp = int((ref_pred & ~ref_true).sum())
    fn = int((~ref_pred & ref_true).sum())
    out = {
        "qwk": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "accuracy": float((pred == y_true).mean()),
        "referable_sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "referable_specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "referable_auc": float("nan"),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(probs.shape[1]))).tolist(),
        "n": int(len(y_true)),
    }
    if 0 < ref_true.sum() < len(ref_true):
        out["referable_auc"] = float(roc_auc_score(ref_true, p_ref))
    return out


def format_metrics(m: dict) -> str:
    return (f"QWK {m['qwk']:.3f} | acc {m['accuracy']:.3f} | "
            f"referable sens {m['referable_sensitivity']:.3f} spec {m['referable_specificity']:.3f} "
            f"AUC {m['referable_auc']:.3f}")
