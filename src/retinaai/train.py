"""Train the DR grading model.

Quick sanity run (~2-3 min on a Kaggle T4):
    python -m retinaai.train --max_samples 600 --epochs 2 --img_size 256 --output_dir outputs/smoke
Full baseline (~15-25 min):
    python -m retinaai.train --epochs 10 --output_dir outputs/baseline
Continue / fine-tune from earlier weights (staged training):
    python -m retinaai.train --init_from outputs/baseline/model_inference.pt --smartphone_aug_p 0.5 ...
Resume an interrupted run:
    python -m retinaai.train --resume outputs/baseline/last.pt --output_dir outputs/baseline
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig, config_to_dict, parse_config
from .data import FundusDataset, build_dataframe, class_weights, ensure_cache, split
from .metrics import compute_metrics, format_metrics
from .model import DEFAULT_HEADS, RetinaModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clean(obj):
    """Make metrics JSON/torch-safe (python floats only)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


@torch.no_grad()
def evaluate(model, loader, device, amp: bool):
    model.eval()
    all_probs, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)["dr"]
        all_probs.append(logits.float().softmax(1).cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_probs)


def run(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # ---- data --------------------------------------------------------
    df = build_dataframe(cfg)
    if cfg.cache_dir:
        t0 = time.time()
        df = ensure_cache(df, cfg.cache_dir, cfg.cache_size)
        print(f"Cache ready ({len(df)} images, {time.time() - t0:.0f}s)")
    train_df, val_df = split(df, cfg.val_fraction, cfg.seed)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    print(f"Train {len(train_df)} | Val {len(val_df)} | "
          f"label counts {np.bincount(train_df['label'], minlength=5).tolist()}")

    train_ds = FundusDataset(train_df, cfg.img_size, True, cfg.preprocess, cfg.smartphone_aug_p)
    val_ds = FundusDataset(val_df, cfg.img_size, False, cfg.preprocess)
    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, cfg.batch_size * 2, shuffle=False, **loader_kw)

    # ---- model -------------------------------------------------------
    model = RetinaModel(cfg.backbone, cfg.pretrained, cfg.dropout, DEFAULT_HEADS).to(device)
    if cfg.init_from:
        state = torch.load(cfg.init_from, map_location="cpu", weights_only=True)["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Initialised from {cfg.init_from} (missing={len(missing)}, unexpected={len(unexpected)})")

    weight = class_weights(train_df["label"]).to(device) if cfg.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_dl)
    warmup = max(1, int(cfg.warmup_epochs * len(train_dl)))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best, history = 0, -float("inf"), []
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)  # own file only
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best_score"], ck.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    def save(path, epoch, metrics, full=True):
        payload = {"format_version": 1, "model": model.state_dict(),
                   "config": config_to_dict(cfg), "heads": DEFAULT_HEADS,
                   "epoch": epoch, "metrics": _clean(metrics)}
        if full:
            payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           scaler=scaler.state_dict(), best_score=best, history=_clean(history))
        torch.save(payload, path)

    # ---- loop --------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x)["dr"], y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if (step + 1) % cfg.log_every == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{len(train_dl)} loss {running / (step + 1):.4f}")

        y_true, probs = evaluate(model, val_dl, device, use_amp)
        m = compute_metrics(y_true, probs)
        m["train_loss"] = running / max(1, len(train_dl))
        m["epoch"] = epoch + 1
        history.append(_clean(m))
        print(f"Epoch {epoch + 1}/{cfg.epochs} ({time.time() - t0:.0f}s) "
              f"loss {m['train_loss']:.4f} | {format_metrics(m)}")

        if m["qwk"] > best:
            best = m["qwk"]
            save(out_dir / "model_inference.pt", epoch, m, full=False)  # small file to share
            np.save(out_dir / "val_probs.npy", probs)
            print(f"  -> new best (QWK {best:.3f}) saved to model_inference.pt")
        save(out_dir / "last.pt", epoch, m, full=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    summary = {"best_qwk": best, "output_dir": str(out_dir),
               "best_epoch_metrics": max(history, key=lambda h: h["qwk"]) if history else None}
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    return summary


def main():
    print(json.dumps(run(parse_config()), indent=2))


if __name__ == "__main__":
    main()
