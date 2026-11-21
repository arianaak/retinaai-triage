"""Does the system "know when it isn't sure" on low-quality images?

Reads the per-image predictions saved by `retinaai.evaluate --save_probs` and applies
the full backend pipeline at each degradation level:
  quality reject -> retake, then review rule (confidence / gap), then T/U safety rules.
Reports how many of the model's mistakes end up with a human (review or retake)
instead of being auto-accepted.

    python scripts/review_analysis.py outputs/robustness_baseline.npz --T 0.15 --U 0.25
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_analysis import triage  # noqa: E402

REJECT = 2  # quality code used by retinaai.evaluate


def analyse(y, probs, quality, T, U, conf, use_quality):
    t = triage(probs, T, U, conf)
    if use_quality:
        t = t.copy()
        t[quality == REJECT] = "retake"
    pred = probs.argmax(1)
    wrong = pred != y
    wrong_ref = (pred >= 2) != (y >= 2)                      # wrong refer / don't-refer
    human = np.isin(t, ["review", "retake"])
    auto = ~human
    return {
        "to human (review+retake)": f"{human.mean():.1%}",
        "retake": f"{(t == 'retake').mean():.1%}",
        "accuracy auto-accepted": f"{(pred[auto] == y[auto]).mean():.1%}",
        "accuracy sent to human": f"{(pred[human] == y[human]).mean():.1%}" if human.any() else "-",
        "grade errors caught": f"{(wrong & human).sum() / max(wrong.sum(), 1):.0%}",
        "referral errors caught": f"{(wrong_ref & human).sum() / max(wrong_ref.sum(), 1):.0%}",
        "missed referrals (auto, routine)": int(((y >= 2) & (t == "routine")).sum()),
        "missed urgent (auto)": int(((y >= 3) & np.isin(t, ["routine", "referable"])).sum()),
        "unnecessary referrals": int(((y <= 1) & np.isin(t, ["referable", "urgent"])).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--U", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.6)
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=False)
    y = d["y"]
    names = {0.0: "clean", 0.5: "moderate", 1.0: "strong"}
    for use_quality in (False, True):
        rows = []
        for s in d["strengths"]:
            r = analyse(y, d[f"probs_{s}"], d[f"quality_{s}"], a.T, a.U, a.conf, use_quality)
            rows.append({"images": names.get(float(s), str(s)), **r})
        title = "WITH quality check (reject -> retake)" if use_quality else "Review rule only (no quality check)"
        print(f"\n{title} | n = {len(y)} | conf < {a.conf} or gap < 0.15 -> review | T={a.T}, U={a.U}")
        print(pd.DataFrame(rows).set_index("images").T.to_string())


if __name__ == "__main__":
    main()
