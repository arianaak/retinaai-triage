"""Tests that need only numpy/OpenCV (no torch): preprocessing, augmentation,
quality, triage, metrics, result contract and the mock predictor."""

import base64

import cv2
import numpy as np

from retinaai.augment import geometric, smartphone_degrade
from retinaai.metrics import compute_metrics
from retinaai.mock import MockPredictor
from retinaai.preprocess import enhance, load_rgb, normalize_chw, to_square_fundus
from retinaai.quality import assess_quality
from retinaai.triage import triage

REQUIRED_KEYS = {"api_version", "mock", "dr", "uncertainty", "quality", "triage",
                 "heatmap_png_base64", "model", "disclaimer"}


def test_preprocess_shapes(fundus):
    sq = to_square_fundus(fundus, 384)
    assert sq.shape == (384, 384, 3) and sq.dtype == np.uint8
    x = normalize_chw(enhance(sq, "ben"))
    assert x.shape == (3, 384, 384) and x.dtype == np.float32


def test_load_from_bytes(fundus):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(fundus, cv2.COLOR_RGB2BGR))
    assert ok
    assert np.array_equal(load_rgb(buf.tobytes()), fundus)


def test_augmentations_keep_shape(fundus):
    rng = np.random.default_rng(0)
    sq = to_square_fundus(fundus, 256)
    for _ in range(20):
        out = smartphone_degrade(geometric(sq, rng), rng)
        assert out.shape == sq.shape and out.dtype == np.uint8


def test_quality_good_vs_degraded(fundus):
    good = assess_quality(fundus)
    assert good["label"] in {"good", "usable"}, good
    blurred = cv2.GaussianBlur(fundus, (0, 0), 12)
    bad = assess_quality(blurred)
    assert bad["score"] < good["score"]
    assert bad["label"] in {"usable", "reject"}


def test_quality_rejects_non_fundus():
    rng = np.random.default_rng(1)
    face_like = np.clip(rng.normal(150, 20, (480, 640, 3)), 0, 255).astype(np.uint8)
    assert assess_quality(face_like)["label"] == "reject"
    assert assess_quality(np.zeros((300, 300, 3), np.uint8))["label"] == "reject"


def test_triage_rules():
    healthy = np.array([0.9, 0.07, 0.02, 0.005, 0.005])
    assert triage(healthy, "good", 0.1, 0.01)["category"] == "routine"
    severe = np.array([0.02, 0.03, 0.15, 0.4, 0.4])
    assert triage(severe, "good", 0.3, 0.05)["category"] == "urgent"
    borderline = np.array([0.3, 0.2, 0.4, 0.05, 0.05])
    assert triage(borderline, "good", 0.4, 0.05)["category"] == "human_review"
    assert triage(severe, "reject", 0.3, 0.05)["category"] == "ungradable"


def test_metrics():
    y = np.array([0, 0, 2, 3, 4, 1])
    probs = np.eye(5)[y] * 0.9 + 0.02
    m = compute_metrics(y, probs)
    assert m["qwk"] > 0.99 and m["referable_sensitivity"] == 1.0


def test_mock_contract(fundus):
    res = MockPredictor().predict(fundus)
    assert REQUIRED_KEYS <= res.keys()
    assert res["mock"] is True
    assert abs(sum(res["dr"]["probabilities"].values()) - 1) < 1e-3
    assert res["triage"]["category"] in {"ungradable", "urgent", "human_review", "referable", "routine"}
    png = base64.b64decode(res["heatmap_png_base64"])
    assert png[:4] == b"\x89PNG"
