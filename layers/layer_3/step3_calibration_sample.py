"""step3_calibration_sample.py
==================================================
Human-calibration instrument for the acceptance threshold.

The perturbation study shows the metric responds to damage; it does not show
that the accept/reject boundary at tau = 0.6 agrees with a human judging "do
these two rows state the same operational rule?". This script makes that check
a one-hour annotation task instead of a missing study.

`sample` draws a stratified, deterministic sample of assigned pairs from the
frozen match audit -- across corpora, across accepted and rejected decisions,
oversampling the near-threshold band where the boundary actually operates --
and writes an annotation sheet holding ONLY the two text blobs, in shuffled
order, with every score and decision withheld. The withheld key is written to
a separate file. An annotator (ideally not the author, ideally two) fills the
`human_same_rule` column with y or n.

`score` joins the filled sheet against the key and reports raw agreement and
Cohen's kappa between the human labels and the metric's tau = 0.6 decisions,
plus the accuracy of every candidate threshold against the human labels, so
the reported operating point can be located relative to the human boundary.

Usage:
    python3 step3_calibration_sample.py sample [--n 60] [--seed 42]
    python3 step3_calibration_sample.py score  <filled_sheet.csv>
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import pandas as pd

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AUDIT_DIR = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "match_audit")
OUT_DIR = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "calibration")

SHEET = os.path.join(OUT_DIR, "calibration_sheet.csv")
KEY = os.path.join(OUT_DIR, "calibration_key.csv")

NEAR_BAND = (0.45, 0.75)   # oversampled: where the boundary actually decides
THRESHOLD = 0.6


def load_assigned_pairs() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(AUDIT_DIR, "*", "match_audit_*.csv"))):
        df = pd.read_csv(path)
        df["corpus"] = os.path.basename(os.path.dirname(path))
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    assigned = rows[rows["decision"].astype(str).str.startswith(("TP", "REJECTED"))].copy()
    # The same GT row meets the same record in several runs; one judgment
    # covers them all, so duplicates are collapsed before sampling.
    assigned = assigned.drop_duplicates(
        subset=["corpus", "gt_text_compared", "pred_text_compared"])
    return assigned


def cmd_sample(n: int, seed: int) -> None:
    pairs = load_assigned_pairs()
    rng = random.Random(seed)

    near = pairs[pairs["combined_score"].between(*NEAR_BAND)]
    far = pairs[~pairs["combined_score"].between(*NEAR_BAND)]

    take_near = min(len(near), n // 2)
    take_far = min(len(far), n - take_near)

    def stratified(df: pd.DataFrame, k: int) -> list[int]:
        chosen: list[int] = []
        pools = [list(g.index)
                 for _, g in df.groupby(["corpus", df["decision"].str[:2]])]
        for pool in pools:
            rng.shuffle(pool)
        while len(chosen) < k and any(pools):
            for pool in pools:
                if pool:
                    chosen.append(pool.pop())
                    if len(chosen) >= k:
                        break
        return chosen

    idx = stratified(near, take_near) + stratified(far, take_far)
    sample = pairs.loc[idx].copy()
    order = list(range(len(sample)))
    rng.shuffle(order)
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "item", [f"P{i+1:03d}" for i in range(len(sample))])

    os.makedirs(OUT_DIR, exist_ok=True)
    sheet = sample[["item", "gt_text_compared", "pred_text_compared"]].copy()
    sheet["human_same_rule"] = ""   # annotator writes y or n
    sheet.to_csv(SHEET, index=False)
    key = sample[["item", "corpus", "gt_id", "pred_id",
                  "combined_score", "decision"]]
    key.to_csv(KEY, index=False)

    n_near = int(sample["combined_score"].between(*NEAR_BAND).sum())
    print(f"[calibration] {len(sample)} pairs sampled "
          f"({n_near} in the near-threshold band {NEAR_BAND})")
    print(f"  annotate : {SHEET}  (fill human_same_rule with y/n; do not open the key)")
    print(f"  key      : {KEY}")


def cmd_score(filled_path: str) -> None:
    sheet = pd.read_csv(filled_path)
    key = pd.read_csv(KEY)
    df = sheet.merge(key, on="item")
    df["human"] = df["human_same_rule"].astype(str).str.strip().str.lower().map(
        {"y": True, "yes": True, "n": False, "no": False})
    df = df.dropna(subset=["human"])
    if df.empty:
        sys.exit("no filled labels found in the sheet")

    metric = df["combined_score"] >= THRESHOLD
    human = df["human"].astype(bool)
    agree = (metric == human).mean()

    # Cohen's kappa
    p_yes = metric.mean() * human.mean()
    p_no = (1 - metric.mean()) * (1 - human.mean())
    pe = p_yes + p_no
    kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")

    print(f"[calibration] {len(df)} labelled pairs")
    print(f"  agreement with tau={THRESHOLD} decisions : {agree:.3f}")
    print(f"  Cohen's kappa                            : {kappa:.3f}")
    print("  accuracy of candidate thresholds against the human labels:")
    for tau in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        acc = ((df["combined_score"] >= tau) == human).mean()
        marker = "  <- reported operating point" if tau == THRESHOLD else ""
        print(f"    tau={tau:.2f}: {acc:.3f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--n", type=int, default=60)
    p_sample.add_argument("--seed", type=int, default=42)
    p_score = sub.add_parser("score")
    p_score.add_argument("filled_sheet")
    args = parser.parse_args()
    if args.cmd == "sample":
        cmd_sample(args.n, args.seed)
    else:
        cmd_score(args.filled_sheet)
