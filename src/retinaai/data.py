"""Dataset loading. Works with any CSV of (image id, label) — APTOS by default,
so DDR / IDRiD / Messidor can be added later by changing the config only."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .augment import geometric, photometric, smartphone_degrade
from .preprocess import enhance, load_rgb, normalize_chw, to_square_fundus


def build_dataframe(cfg) -> pd.DataFrame:
    df = pd.read_csv(Path(cfg.data_dir) / cfg.csv_name)
    img_dir = Path(cfg.data_dir) / cfg.image_dir
    df = pd.DataFrame({
        "id": df[cfg.id_col].astype(str),
        "path": [str(img_dir / f"{i}{cfg.image_ext}") for i in df[cfg.id_col].astype(str)],
        "label": df[cfg.label_col].astype(int),
    })
    if cfg.max_samples and cfg.max_samples < len(df):
        df, _ = train_test_split(df, train_size=cfg.max_samples,
                                 stratify=df["label"], random_state=cfg.seed)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, val_fraction: float, seed: int):
    tr, va = train_test_split(df, test_size=val_fraction, stratify=df["label"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def _cache_one(args):
    src, dst, size = args
    if not os.path.exists(dst):
        img = to_square_fundus(load_rgb(src), size)
        cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return dst


def ensure_cache(df: pd.DataFrame, cache_dir: str, size: int, workers: int = 8) -> pd.DataFrame:
    """Write cropped+resized copies once; later epochs/runs read small files."""
    out_dir = Path(cache_dir) / str(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(out_dir / f"{i}.png"), size) for i, p in zip(df["id"], df["path"])]
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(_cache_one, jobs))
    df = df.copy()
    df["path"] = paths
    return df


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool,
                 preprocess: str = "ben", smartphone_p: float = 0.0):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.img_size = img_size
        self.train = train
        self.preprocess = preprocess
        self.smartphone_p = smartphone_p

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        img = to_square_fundus(load_rgb(self.paths[idx]), self.img_size)
        if self.train:
            img = geometric(img, rng)
            if rng.random() < self.smartphone_p:
                img = smartphone_degrade(img, rng)
            else:
                img = photometric(img, rng)
        img = enhance(img, self.preprocess)
        return torch.from_numpy(normalize_chw(img)), self.labels[idx]


def class_weights(labels, num_classes: int = 5) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)
