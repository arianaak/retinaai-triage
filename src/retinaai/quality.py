"""Heuristic image-quality assessment (stage 1).

Rule-based so it works before any quality model is trained. Thresholds are
PLACEHOLDERS: calibrate them with `scripts/quality_report.py` on real data
(clean APTOS images vs. degraded ones). Stage 3 replaces/augments this with
a classifier trained on EyeQ (Good / Usable / Reject).
numpy/OpenCV only (no torch).
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import to_square_fundus

ANALYSIS_SIZE = 512

# --- thresholds (CALIBRATE ON REAL DATA) ---------------------------------
THRESHOLDS = {
    "sharpness_reject": 2.0,     # variance of Laplacian inside the fundus
    "sharpness_usable": 6.0,
    "brightness_min": 35.0,      # mean grey level inside the fundus
    "brightness_max": 210.0,
    "overexposed_usable": 0.05,  # fraction of near-white pixels (glare)
    "overexposed_reject": 0.20,
    "underexposed_reject": 0.40,  # fraction of very dark pixels inside fundus
    "coverage_usable": 0.45,     # fundus area / image area after cropping
    "corner_dark_max": 40.0,     # fundus photos have dark corners
    "red_ratio_min": 1.05,       # mean R / mean G inside the fundus
}


def quality_metrics(img_rgb: np.ndarray) -> dict:
    img = to_square_fundus(img_rgb, ANALYSIS_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > 10).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))  # avoid the rim edge
    inside = mask.astype(bool)
    n = int(inside.sum())
    if n < 0.05 * mask.size:
        return {"valid": False, "coverage": n / mask.size}

    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    g_in = gray[inside].astype(np.float32)
    rgb_in = img[inside].astype(np.float32)
    c = ANALYSIS_SIZE // 10
    corners = np.concatenate([gray[:c, :c].ravel(), gray[:c, -c:].ravel(),
                              gray[-c:, :c].ravel(), gray[-c:, -c:].ravel()])
    return {
        "valid": True,
        "sharpness": float(lap[inside].var()),
        "brightness": float(g_in.mean()),
        "contrast": float(g_in.std()),
        "overexposed_frac": float((g_in > 235).mean()),
        "underexposed_frac": float((g_in < 20).mean()),
        "coverage": n / mask.size,
        "corner_brightness": float(corners.mean()),
        "red_ratio": float(rgb_in[:, 0].mean() / (rgb_in[:, 1].mean() + 1e-6)),
    }


def assess_quality(img_rgb: np.ndarray, thresholds: dict | None = None) -> dict:
    """Return {'label': 'good'|'usable'|'reject', 'score': 0..1, 'reasons': [...], 'metrics': {...}}"""
    t = {**THRESHOLDS, **(thresholds or {})}
    m = quality_metrics(img_rgb)
    reasons_reject, reasons_usable = [], []

    if not m.get("valid"):
        return {"label": "reject", "score": 0.0,
                "reasons": ["No retinal field detected (image too dark or empty)."], "metrics": m}

    if m["corner_brightness"] > t["corner_dark_max"] or m["red_ratio"] < t["red_ratio_min"]:
        reasons_reject.append("Does not look like a fundus photograph. "
                              "A regular camera photo of the eye cannot show the retina.")
    if m["sharpness"] < t["sharpness_reject"]:
        reasons_reject.append("Image is too blurry / out of focus.")
    elif m["sharpness"] < t["sharpness_usable"]:
        reasons_usable.append("Image is somewhat blurry.")
    if m["brightness"] < t["brightness_min"] or m["underexposed_frac"] > t["underexposed_reject"]:
        reasons_reject.append("Image is too dark (underexposed or small pupil).")
    if m["brightness"] > t["brightness_max"] or m["overexposed_frac"] > t["overexposed_reject"]:
        reasons_reject.append("Image is overexposed / strong glare.")
    elif m["overexposed_frac"] > t["overexposed_usable"]:
        reasons_usable.append("Some glare or reflections present.")
    if m["coverage"] < t["coverage_usable"]:
        reasons_usable.append("Small field of view; peripheral lesions may be missed.")

    # crude 0..1 score for display / ranking
    s_sharp = np.clip((m["sharpness"] - t["sharpness_reject"]) /
                      (2 * t["sharpness_usable"] - t["sharpness_reject"]), 0, 1)
    s_expo = 1.0 - np.clip(m["overexposed_frac"] / t["overexposed_reject"], 0, 1)
    s_dark = 1.0 - np.clip(m["underexposed_frac"] / t["underexposed_reject"], 0, 1)
    s_cov = np.clip(m["coverage"] / 0.75, 0, 1)
    score = float(np.round(0.4 * s_sharp + 0.2 * s_expo + 0.2 * s_dark + 0.2 * s_cov, 3))

    if reasons_reject:
        label, score = "reject", min(score, 0.3)
    elif reasons_usable:
        label = "usable"
    else:
        label = "good"
    return {"label": label, "score": score,
            "reasons": reasons_reject + reasons_usable, "metrics": m}
