"""Grad-CAM attention maps (no extra dependencies).

Target = expected DR grade (sum_k k * p_k), so the map shows regions that
pushed the prediction towards MORE severe disease. These are coarse
"where the model looked" maps — not proof of reasoning.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .preprocess import circle_mask


def grad_cam(model, x, head: str = "dr") -> np.ndarray:
    import torch  # local import: overlay/to_png_base64 stay usable without torch (mock mode)

    model.eval()
    with torch.enable_grad():
        out, fmap = model(x, return_features=True)
        fmap.retain_grad()
        probs = out[head].float().softmax(1)
        grades = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
        score = (probs * grades).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * fmap).sum(1))[0]
    cam = cam.detach().float().cpu().numpy()
    cam -= cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    size = rgb.shape[0]
    cam = cv2.resize(cam, (rgb.shape[1], size), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    mask = circle_mask(size, 0.49)[..., None]
    blended = (rgb * (1 - alpha) + heat * alpha) * mask + rgb * (1 - mask)
    return blended.astype(np.uint8)


def to_png_base64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
