"""Simulate the backend's triage rules on saved validation predictions (no torch needed).

Rule order (first match wins):
  review   if top1 < CONF or top1 - top2 < GAP
  referable if predicted 0-1 and p_ref = p2+p3+p4 >= T        (safety rule)
  urgent    if predicted 2   and p_urgent = p3+p4 >= U        (safety rule)
  grade 0-1 routine | 2 referable | 3-4 urgent

    python scripts/triage_analysis.py --run outputs/baseline --T 0.15 --U 0.25
"""

import argparse

import numpy as np
import pandas as pd

LEVELS = ["urgent", "review", "referable", "retake", "routine"]


def triage(probs, T=None, U=None, conf=0.6, gap_t=0.15):
    pred = probs.argmax(1)
    top = np.sort(probs, 1)
    top1, gap = top[:, -1], top[:, -1] - top[:, -2]
    p_ref, p_urg = probs[:, 2:].sum(1), probs[:, 3:].sum(1)
    out = np.where(pred <= 1, "routine", np.where(pred == 2, "referable", "urgent")).astype(object)
    if U is not None:
        out[(pred == 2) & (p_urg >= U)] = "urgent"
    if T is not None:
        out[(pred <= 1) & (p_ref >= T)] = "referable"
    out[(top1 < conf) | (gap < gap_t)] = "review"
    return out


def summary(y, t):
    return {
        "missed referrals": int(((y >= 2) & (t == "routine")).sum()),
        "missed urgent": int(((y >= 3) & ~np.isin(t, ["urgent", "review"])).sum()),
        "unnecessary referrals": int(((y <= 1) & np.isin(t, ["referable", "urgent"])).sum()),
        "false urgent": int(((y <= 2) & (t == "urgent")).sum()),
        "review rate": f"{(t == 'review').mean():.1%}",
        "referred": f"{np.isin(t, ['referable', 'urgent']).mean():.1%}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/baseline")
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--U", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.6)
    a = ap.parse_args()

    probs = np.load(f"{a.run}/val_probs.npy")
    y = pd.read_csv(f"{a.run}/val_split.csv", dtype={"id": str}).label.values

    print("T sweep (no U):")
    print(pd.DataFrame([{"T": T, **summary(y, triage(probs, T=T, conf=a.conf))}
                        for T in [None, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]]).to_string(index=False))
    print(f"\nU sweep (T={a.T}):")
    print(pd.DataFrame([{"U": U, **summary(y, triage(probs, T=a.T, U=U, conf=a.conf))}
                        for U in [None, 0.20, 0.25, 0.30, 0.35, 0.40]]).to_string(index=False))
    for name, t in [("today's rules", triage(probs, conf=a.conf)),
                    (f"T={a.T}, U={a.U}", triage(probs, a.T, a.U, a.conf))]:
        tab = pd.crosstab(pd.Series(y, name="true grade"), pd.Series(t, name="triage"))
        print(f"\n{name}:\n{tab.reindex(columns=LEVELS, fill_value=0).to_string()}\n{summary(y, t)}")


if __name__ == "__main__":
    main()
