"""Model/inference smoke tests. Skipped automatically where torch/timm are missing."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from retinaai.model import DEFAULT_HEADS, RetinaModel  # noqa: E402


def test_forward_and_gradcam():
    from retinaai.explain import grad_cam
    model = RetinaModel("efficientnet_b0", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out, fmap = model(x, return_features=True)
    assert out["dr"].shape == (2, 5) and fmap.ndim == 4
    cam = grad_cam(model, x[:1])
    assert cam.ndim == 2 and 0 <= cam.min() and cam.max() <= 1


def test_checkpoint_roundtrip(tmp_path, fundus):
    from retinaai.config import TrainConfig, config_to_dict
    from retinaai.predict import Predictor
    cfg = TrainConfig(img_size=224, pretrained=False)
    model = RetinaModel(cfg.backbone, False, cfg.dropout)
    path = tmp_path / "m.pt"
    torch.save({"format_version": 1, "model": model.state_dict(), "config": config_to_dict(cfg),
                "heads": DEFAULT_HEADS, "epoch": 0, "metrics": {"qwk": 0.0}}, path)
    res = Predictor(str(path), device="cpu", mc_samples=2).predict(fundus)
    assert res["mock"] is False and res["heatmap_png_base64"]
