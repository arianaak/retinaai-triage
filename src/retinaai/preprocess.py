"""Image loading and preprocessing shared by training AND inference.

Keeping one implementation guarantees the model sees identical preprocessing
in the notebook and in the deployed API. numpy/OpenCV only (no torch).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(source) -> np.ndarray:
    """Load an image from a path, raw bytes, a PIL image or an RGB numpy array."""
    if isinstance(source, np.ndarray):
        img = source
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    if hasattr(source, "convert"):  # PIL.Image
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(source, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(source), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread(str(Path(source)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (unsupported or corrupt file).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_to_fundus(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Remove the black frame around the circular fundus."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if mask.mean() < 0.01:  # nearly black image: leave untouched
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=img.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y: y + h, x: x + w] = img
    return out


def to_square_fundus(img: np.ndarray, size: int) -> np.ndarray:
    """crop black border -> pad to square -> resize. Output uint8 RGB (size, size, 3)."""
    img = pad_to_square(crop_to_fundus(img))
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def circle_mask(size: int, radius_frac: float = 0.48) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * radius_frac), 1, -1)
    return mask


def ben_graham(img: np.ndarray) -> np.ndarray:
    """Local-contrast enhancement popularised in the 2015 Kaggle DR competition:
    subtract a heavy Gaussian blur so lesions/vessels stand out from uneven lighting."""
    size = img.shape[0]
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=size / 50)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    mask = circle_mask(size, 0.48)[..., None]
    return (out * mask + 128 * (1 - mask)).astype(np.uint8)


def enhance(img: np.ndarray, method: str = "ben") -> np.ndarray:
    if method == "ben":
        return ben_graham(img)
    if method == "none":
        return img
    raise ValueError(f"Unknown preprocess method: {method}")


def normalize_chw(img: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB -> float32 CHW, ImageNet-normalised."""
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))
