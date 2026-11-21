"""Calibrate the heuristic quality thresholds on real data.

Computes quality metrics for a folder of images, optionally also for
artificially degraded copies, and writes a CSV + percentile summary.
Pick thresholds so that clean images pass and degraded ones are flagged.

    python scripts/quality_report.py --images /kaggle/input/aptos2019-blindness-detection/train_images \
        --limit 300 --degrade --out outputs/quality_report.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinaai.augment import smartphone_degrade  # noqa: E402
from retinaai.preprocess import load_rgb, to_square_fundus  # noqa: E402
from retinaai.quality import assess_quality  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--degrade", action="store_true", help="also score smartphone-degraded copies")
ap.add_argument("--out", default="quality_report.csv")
args = ap.parse_args()

paths = sorted(p for p in Path(args.images).iterdir()
               if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})[: args.limit]
rng = np.random.default_rng(0)
rows = []
for p in paths:
    img = load_rgb(p)
    variants = [("clean", img)]
    if args.degrade:
        variants.append(("degraded", smartphone_degrade(to_square_fundus(img, 512), rng)))
    for kind, im in variants:
        q = assess_quality(im)
        rows.append({"file": p.name, "kind": kind, "label": q["label"], "score": q["score"],
                     **{k: v for k, v in q["metrics"].items() if k != "valid"}})

df = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(df.groupby("kind")["label"].value_counts().unstack(fill_value=0))
cols = ["sharpness", "brightness", "overexposed_frac", "underexposed_frac", "coverage",
        "corner_brightness", "red_ratio"]
print(df.groupby("kind")[cols].quantile([0.05, 0.5, 0.95]).round(3))
