"""Training augmentations, including simulated smartphone / low-quality capture.

`smartphone_degrade` approximates what smartphone fundus adapters and poor
acquisitions produce: narrow field of view, defocus, uneven illumination,
glare, colour cast, sensor noise and JPEG compression.
It is a PROXY for robustness — not a substitute for validating on real
smartphone images (e.g. mBRSET).
"""

from __future__ import annotations

import cv2
import numpy as np


def geometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    h, w = img.shape[:2]
    angle = rng.uniform(-180, 180)
    scale = rng.uniform(0.95, 1.10)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h),
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)        # contrast / brightness
    gamma = rng.uniform(0.85, 1.15)
    x = 255.0 * np.power(np.clip(x, 0, 255) / 255.0, gamma)
    return np.clip(x, 0, 255).astype(np.uint8)


def smartphone_degrade(img: np.ndarray, rng: np.random.Generator,
                       strength: float = 1.0) -> np.ndarray:
    """Apply a random subset of smartphone-like degradations. `strength` in (0, 1]."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    fov = np.ones((h, w), dtype=np.float32)

    # 1) narrow field of view (adapters often see ~5-20 deg instead of ~45 deg)
    if rng.random() < 0.7 * strength:
        frac = rng.uniform(0.55, 0.85)
        cs = int(w * frac)
        max_off = (w - cs) // 2
        cx = w // 2 + int(rng.uniform(-0.5, 0.5) * max_off)
        cy = h // 2 + int(rng.uniform(-0.5, 0.5) * max_off)
        x0 = int(np.clip(cx - cs // 2, 0, w - cs))
        y0 = int(np.clip(cy - cs // 2, 0, h - cs))
        out = cv2.resize(out[y0: y0 + cs, x0: x0 + cs], (w, h))
        fov = np.zeros((h, w), dtype=np.float32)
        cv2.circle(fov, (w // 2, h // 2), int(w * rng.uniform(0.40, 0.49)), 1.0, -1)

    # 2) defocus / motion-free blur
    if rng.random() < 0.6 * strength:
        sigma = rng.uniform(0.5, 2.5) * strength * (w / 384)
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=max(sigma, 0.1))

    # 3) uneven illumination
    if rng.random() < 0.5 * strength:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        theta = rng.uniform(0, 2 * np.pi)
        ramp = ((xx / w - 0.5) * np.cos(theta) + (yy / h - 0.5) * np.sin(theta))
        out *= (1.0 + rng.uniform(0.2, 0.6) * strength * ramp)[..., None]

    # 4) glare / reflections
    if rng.random() < 0.4 * strength:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for _ in range(int(rng.integers(1, 3))):
            gx, gy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
            s = rng.uniform(0.03, 0.10) * w
            blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * s * s))
            out += (rng.uniform(80, 200) * strength * blob)[..., None]

    # 5) colour cast (different sensor / white balance)
    if rng.random() < 0.6 * strength:
        out *= rng.uniform(1 - 0.15 * strength, 1 + 0.15 * strength, size=3)

    # 6) sensor noise
    if rng.random() < 0.5 * strength:
        out += rng.normal(0, rng.uniform(2, 10) * strength, size=out.shape)

    out = np.clip(out * fov[..., None], 0, 255).astype(np.uint8)

    # 7) JPEG compression
    if rng.random() < 0.5 * strength:
        q = int(rng.uniform(30, 80))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out
