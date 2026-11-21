import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_synthetic_fundus(h=600, w=800, seed=0):
    """Crude fundus-like image: red-orange disc on black, vessels, optic disc, texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    cy, cx, r = h // 2, w // 2, int(h * 0.47)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r
    inside = d < 1
    shade = np.clip(1.0 - 0.45 * d ** 2, 0, 1)
    for c, base in enumerate((190, 85, 40)):
        img[..., c] = base * shade
    # optic disc
    cv2.circle(img, (cx + r // 3, cy), r // 9, (240, 200, 140), -1)
    # vessels
    for k in range(10):
        ang = rng.uniform(0, 2 * np.pi)
        end = (int(cx + r // 3 + np.cos(ang) * r), int(cy + np.sin(ang) * r))
        cv2.line(img, (cx + r // 3, cy), end, (120, 30, 20), int(rng.integers(2, 6)))
    img += rng.normal(0, 6, img.shape)  # fine texture
    img[~inside] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def fundus():
    return make_synthetic_fundus()
