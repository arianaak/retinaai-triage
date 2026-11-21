"""Robustness evaluation: how does a model perform as image quality drops?

Runs a checkpoint over a split (default: the validation split saved during training)
at several levels of simulated smartphone / low-quality degradation, and reports
grading metrics, referral metrics and how often the quality check rejects the image.
Degradation is deterministic per (image, strength, seed), so different models are
compared on exactly the same degraded images.

    python -m retinaai.evaluate --checkpoint outputs/baseline/model_inference.pt \
        --split_csv outputs/baseline/val_split.csv --data_dir $DATA \
        --out outputs/robustness_baseline.json --examples_png outputs/degradation_examples.png \
        --save_probs outputs/robustness_baseline.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import smartphone_degrade
from .metrics import compute_metrics
from .model import load_inference_model
from .preprocess import enhance, load_rgb, normalize_chw, to_square_fundus
from .quality import assess_quality

QUALITY_CODES = {"good": 0, "usable": 1, "reject": 2}


class DegradedSet(Dataset):
    def __init__(self, df, data_dir, image_dir, ext, cfg, strength, seed):
        self.ids = df["id"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.data_dir, self.image_dir, self.ext = data_dir, image_dir, ext
        self.img_size, self.method = cfg["img_size"], cfg["preprocess"]
        self.cache = Path(cfg["cache_dir"]) / str(cfg["cache_size"]) if cfg.get("cache_dir") else None
        self.cache_size = cfg.get("cache_size") if cfg.get("cache_dir") else None
        self.strength, self.seed = strength, seed

    def __len__(self):
        return len(self.ids)

    def square(self, i):
        cached = self.cache / f"{self.ids[i]}.png" if self.cache else None
        if cached is not None and cached.exists():
            img = load_rgb(cached)
        else:
            img = load_rgb(Path(self.data_dir) / self.image_dir / f"{self.ids[i]}{self.ext}")
            if self.cache_size:
                img = to_square_fundus(img, self.cache_size)
        sq = to_square_fundus(img, self.img_size)
        if self.strength > 0:
            rng = np.random.default_rng([self.seed, i, int(self.strength * 1000)])
            sq = smartphone_degrade(sq, rng, self.strength)
        return sq

    def __getitem__(self, i):
        sq = self.square(i)
        q = QUALITY_CODES[assess_quality(sq)["label"]]
        return torch.from_numpy(normalize_chw(enhance(sq, self.method))), self.labels[i], q


@torch.no_grad()
def run_strength(model, ds, device, batch_size, workers):
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=workers)
    probs, ys, qs = [], [], []
    for x, y, q in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))["dr"]
        probs.append(logits.float().softmax(1).cpu().numpy())
        ys.append(y.numpy()); qs.append(q.numpy())
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(qs)


def summarize(y, probs, q, ref_t=0.15):
    allm = compute_metrics(y, probs)
    p_ref = probs[:, 2:].sum(1)
    ref = y >= 2
    acc = q != QUALITY_CODES["reject"]
    row = {
        "qwk": allm["qwk"], "accuracy": allm["accuracy"], "referable_auc": allm["referable_auc"],
        "sens@0.5": allm["referable_sensitivity"], "spec@0.5": allm["referable_specificity"],
        f"sens@{ref_t}": float(((p_ref >= ref_t) & ref).sum() / max(ref.sum(), 1)),
        f"spec@{ref_t}": float(((p_ref < ref_t) & ~ref).sum() / max((~ref).sum(), 1)),
        "quality_reject_rate": float((~acc).mean()),
        "quality_usable_rate": float((q == QUALITY_CODES["usable"]).mean()),
    }
    if acc.sum() > 10 and 0 < ref[acc].sum() < acc.sum():
        m = compute_metrics(y[acc], probs[acc])
        row["qwk_on_accepted"] = m["qwk"]
        row["auc_on_accepted"] = m["referable_auc"]
    return row


def save_examples(ds_factory, strengths, path, n_images=3):
    tiles = []
    for i in range(n_images):
        row = [ds_factory(s).square(i) for s in strengths]
        tiles.append(np.concatenate(row, axis=1))
    grid = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--image_dir", default="train_images")
    ap.add_argument("--image_ext", default=".png")
    ap.add_argument("--strengths", default="0,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--examples_png", default=None)
    ap.add_argument("--save_probs", default=None, help="save per-image predictions (.npz) for later analysis")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_inference_model(a.checkpoint, device)
    cfg = ckpt["config"]
    df = pd.read_csv(a.split_csv, dtype={"id": str})
    if a.max_images:
        df = df.head(a.max_images)
    strengths = [float(s) for s in a.strengths.split(",")]

    def make(s):
        return DegradedSet(df, a.data_dir, a.image_dir, a.image_ext, cfg, s, a.seed)

    results, saved = {}, {}
    for s in strengths:
        y, probs, q = run_strength(model, make(s), device, a.batch_size, a.num_workers)
        results[str(s)] = summarize(y, probs, q)
        saved[f"probs_{s}"], saved[f"quality_{s}"], saved["y"] = probs, q, y
        print(f"strength {s}: " + " | ".join(f"{k} {v:.3f}" for k, v in results[str(s)].items()), flush=True)

    table = pd.DataFrame(results).T
    table.index.name = "degradation strength"
    print("\n" + table.round(3).to_string())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"checkpoint": os.path.abspath(a.checkpoint),
                                           "n_images": len(df), "results": results}, indent=2))
    if a.save_probs:
        np.savez(a.save_probs, ids=df["id"].astype(str).values, strengths=np.array(strengths), **saved)
        print(f"per-image predictions saved to {a.save_probs}")
    if a.examples_png:
        save_examples(make, strengths, a.examples_png)
        print(f"example grid saved to {a.examples_png}")


if __name__ == "__main__":
    main()
