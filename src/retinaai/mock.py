"""Mock predictor: same output format as the real one, no torch / no weights.

Lets the backend and frontend be built and tested BEFORE the model is trained.
Quality assessment is real; DR probabilities are fake but deterministic per image.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .explain import to_png_base64
from .preprocess import load_rgb, to_square_fundus
from .quality import assess_quality
from .result import build_result


class MockPredictor:
    def __init__(self, *_, **__):
        self.model_info = {"backbone": "mock", "img_size": 384, "trained_epoch": 0, "val_metrics": {}}

    def predict(self, image, return_heatmap: bool = True) -> dict:
        rgb = load_rgb(image)
        seed = int(hashlib.md5(rgb[::16, ::16].tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        probs = rng.dirichlet(np.ones(5) * 0.7)
        heatmap = None
        if return_heatmap:
            sq = to_square_fundus(rgb, 384)
            cam = cv2.GaussianBlur(rng.random((12, 12)).astype(np.float32), (0, 0), 2)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
            from .explain import overlay
            heatmap = to_png_base64(overlay(sq, cam))
        return build_result(probs, assess_quality(rgb), float(rng.uniform(0, 0.2)),
                            heatmap, self.model_info, mock=True)
