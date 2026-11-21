"""Training configuration. Every field can be overridden from the command line,
e.g.  python -m retinaai.train --epochs 5 --img_size 256 --max_samples 600
"""

import argparse
import typing
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class TrainConfig:
    # ---- data ----------------------------------------------------------
    data_dir: str = "/kaggle/input/aptos2019-blindness-detection"
    csv_name: str = "train.csv"
    image_dir: str = "train_images"
    image_ext: str = ".png"          # appended to the id column ("" if ids already have it)
    id_col: str = "id_code"
    label_col: str = "diagnosis"
    val_fraction: float = 0.2
    max_samples: Optional[int] = None  # use a stratified subset for quick experiments
    cache_dir: Optional[str] = "/kaggle/working/cache"  # resized copies -> much faster epochs
    cache_size: int = 512

    # ---- model ---------------------------------------------------------
    backbone: str = "efficientnet_b0"  # any timm CNN, e.g. efficientnet_b3, resnet50
    pretrained: bool = True
    dropout: float = 0.3
    img_size: int = 384
    preprocess: str = "ben"            # "ben" (Ben Graham contrast enhancement) or "none"

    # ---- training ------------------------------------------------------
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 0.5
    label_smoothing: float = 0.05
    class_weighting: bool = True       # sqrt-inverse-frequency weights (APTOS is imbalanced)
    smartphone_aug_p: float = 0.3      # probability of simulated smartphone/low-quality degradation
    amp: bool = True
    num_workers: int = 2
    seed: int = 42

    # ---- bookkeeping ---------------------------------------------------
    output_dir: str = "outputs/run"
    resume: Optional[str] = None       # continue an interrupted run (last.pt)
    init_from: Optional[str] = None    # start from previous weights (staged / transfer training)
    log_every: int = 20


def _unwrap(tp):
    """Optional[int] -> int"""
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return args[0]
    return tp


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}


def parse_config(argv=None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the RetinaAI DR model")
    for f in fields(TrainConfig):
        tp = _unwrap(f.type)
        if tp is bool:
            parser.add_argument(f"--{f.name}", type=_str2bool, default=f.default)
        else:
            def conv(v, _tp=tp):
                return None if str(v).lower() == "none" else _tp(v)
            parser.add_argument(f"--{f.name}", type=conv, default=f.default)
    return TrainConfig(**vars(parser.parse_args(argv)))


def config_to_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
