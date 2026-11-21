"""Model: pretrained timm backbone + one linear head per task.

Multi-head from day one, so later stages can add e.g. a quality head (EyeQ)
or multi-disease heads (RFMiD/ODIR) without changing the inference code.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

DEFAULT_HEADS = {"dr": 5}


class RetinaModel(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 dropout: float = 0.3, heads: dict | None = None):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        nf = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)  # also used for MC-dropout uncertainty
        self.heads = nn.ModuleDict({k: nn.Linear(nf, n) for k, n in (heads or DEFAULT_HEADS).items()})

    def forward(self, x, return_features: bool = False):
        fmap = self.backbone.forward_features(x)              # (B, C, H, W) — used by Grad-CAM
        feat = self.backbone.forward_head(fmap, pre_logits=True)
        feat = self.dropout(feat)
        out = {k: head(feat) for k, head in self.heads.items()}
        return (out, fmap) if return_features else out


def enable_mc_dropout(model: nn.Module):
    """Eval mode everywhere (BatchNorm frozen) but dropout active."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def load_inference_model(path: str, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = RetinaModel(cfg["backbone"], pretrained=False, dropout=cfg["dropout"],
                        heads=ckpt.get("heads", DEFAULT_HEADS))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
