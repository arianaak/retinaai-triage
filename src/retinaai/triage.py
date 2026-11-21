"""Turn model probabilities + quality + uncertainty into a triage category.

Rules are deliberately simple and transparent. Thresholds are prototype
choices, not validated clinical cut-offs.
numpy only (no torch).
"""

from __future__ import annotations

import numpy as np

TRIAGE_RULES = {
    "referable_threshold": 0.5,   # P(grade >= 2)
    "urgent_threshold": 0.5,      # P(grade >= 3)
    "uncertain_band": (0.30, 0.70),  # P(referable) in this band -> human review
    "max_entropy": 0.60,          # normalised predictive entropy above this -> human review
    "max_std": 0.15,              # std of P(referable) across MC samples above this -> human review
}

CATEGORIES = {
    "ungradable": {"priority": 0, "action": "Retake the image; result not reliable."},
    "urgent": {"priority": 1, "action": "Possible severe/proliferative DR: prioritise specialist review."},
    "human_review": {"priority": 2, "action": "Model uncertain: needs review by a clinician."},
    "referable": {"priority": 3, "action": "Possible referable DR: refer for ophthalmology evaluation."},
    "routine": {"priority": 4, "action": "No referable DR detected: routine rescreening per local guidelines."},
}


def triage(probs, quality_label: str, entropy: float, p_ref_std: float,
           rules: dict | None = None) -> dict:
    r = {**TRIAGE_RULES, **(rules or {})}
    probs = np.asarray(probs, dtype=float)
    p_ref = float(probs[2:].sum())
    p_urgent = float(probs[3:].sum())
    reasons = []

    if quality_label == "reject":
        cat = "ungradable"
        reasons.append("Image quality insufficient for analysis.")
    elif p_urgent >= r["urgent_threshold"]:
        cat = "urgent"
        reasons.append(f"P(severe or proliferative) = {p_urgent:.2f}")
    else:
        lo, hi = r["uncertain_band"]
        if lo <= p_ref <= hi:
            reasons.append(f"P(referable) = {p_ref:.2f} is borderline.")
        if entropy > r["max_entropy"]:
            reasons.append("High predictive uncertainty.")
        if p_ref_std > r["max_std"]:
            reasons.append("Unstable prediction across repeated passes.")
        if quality_label == "usable" and p_ref >= lo:
            reasons.append("Reduced image quality with possible findings.")
        if reasons:
            cat = "human_review"
        elif p_ref >= r["referable_threshold"]:
            cat = "referable"
            reasons.append(f"P(referable) = {p_ref:.2f}")
        else:
            cat = "routine"
            reasons.append(f"P(referable) = {p_ref:.2f}")

    return {"category": cat, **CATEGORIES[cat],
            "p_referable": round(p_ref, 4), "p_urgent": round(p_urgent, 4),
            "reasons": reasons}
