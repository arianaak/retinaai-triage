"""Builds the JSON result returned to the backend/frontend.
Single source of truth for the API contract (see docs/API_CONTRACT.md).
numpy only (no torch) so the mock predictor can share it."""

from __future__ import annotations

import numpy as np

from . import DISCLAIMER, GRADE_NAMES, __version__
from .triage import triage


def predictive_entropy(probs) -> float:
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def build_result(probs, quality: dict, p_ref_std: float, heatmap_b64: str | None,
                 model_info: dict, mock: bool = False) -> dict:
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    entropy = predictive_entropy(probs)
    grade = int(probs.argmax())
    tri = triage(probs, quality["label"], entropy, p_ref_std)
    return {
        "api_version": "1",
        "package_version": __version__,
        "mock": mock,
        "dr": {
            "grade": grade,
            "grade_name": GRADE_NAMES[grade],
            "probabilities": {name: round(float(p), 4) for name, p in zip(GRADE_NAMES, probs)},
            "expected_grade": round(float((probs * np.arange(len(probs))).sum()), 3),
        },
        "uncertainty": {
            "entropy": round(entropy, 4),          # 0 = certain, 1 = uniform
            "p_referable_std": round(float(p_ref_std), 4),
            "confidence": round(float(probs.max()), 4),
        },
        "quality": {k: quality[k] for k in ("label", "score", "reasons")},
        "triage": tri,
        "heatmap_png_base64": heatmap_b64,
        "model": model_info,
        "disclaimer": DISCLAIMER,
    }
