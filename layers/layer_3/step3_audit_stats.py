"""step3_audit_stats.py
==================================================
Aggregate statistics over the per-record audit trail.

The audit trail records, for every assigned pair, the best score obtainable
from any OTHER record (runner_up_score) and the margin over it. Aggregated,
those two columns say how decisive the optimal assignment's pairings are:
a row whose margin is non-negative received its own best-scoring record,
while a negative margin means the global optimum handed that record to a
different row. Pairs within +/-0.05 of their runner-up are close calls and
should be read as such rather than as confident matches.

Nothing here re-scores anything: every figure is a count over columns the
evaluator already wrote, so this table cannot drift from the audit files.

Writes step3_results/audit_trail_stats.csv.

Usage:
    python3 step3_audit_stats.py
"""

import glob
import os
import sys

import pandas as pd

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AUDIT_DIR = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "match_audit")
OUT_PATH = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "audit_trail_stats.csv")

CLOSE_CALL_MARGIN = 0.05
# Decisions carry an explanatory suffix ("TP (counted as found)"), so they are
# matched on their prefix.
ASSIGNED_PREFIXES = ("TP", "REJECTED")


def main() -> None:
    paths = sorted(glob.glob(os.path.join(AUDIT_DIR, "*", "match_audit_*.csv")))
    if not paths:
        sys.exit(f"no audit files under {AUDIT_DIR}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["corpus"] = os.path.basename(os.path.dirname(path))
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)

    decision = rows["decision"].astype(str)
    assigned = rows[decision.str.startswith(ASSIGNED_PREFIXES)].copy()
    margins = pd.to_numeric(assigned["margin_over_runner_up"], errors="coerce")

    n = len(assigned)
    own_best = int((margins >= 0).sum())
    yielded = int((margins < 0).sum())
    close = int((margins.abs() <= CLOSE_CALL_MARGIN).sum())

    summary = pd.DataFrame([{
        "audit_files": len(paths),
        "total_audit_rows": len(rows),
        "assigned_pairs": n,
        "tp_pairs": int(decision.str.startswith("TP").sum()),
        "rejected_pairs": int(decision.str.startswith("REJECTED").sum()),
        "unpaired_gt_rows": int(decision.str.startswith("UNPAIRED GT").sum()),
        "unpaired_records": int(decision.str.startswith("UNPAIRED RECORD").sum()),
        "own_best_pairs": own_best,
        "own_best_frac": round(own_best / n, 3),
        "yielded_to_optimum_pairs": yielded,
        "yielded_frac": round(yielded / n, 3),
        f"within_{CLOSE_CALL_MARGIN}_of_runner_up": close,
        "close_call_frac": round(close / n, 3),
    }])
    summary.to_csv(OUT_PATH, index=False)

    print(f"[audit-stats] {len(paths)} audit files, {n} assigned pairs")
    print(f"  own best record : {own_best} ({own_best / n:.1%})")
    print(f"  yielded to optimum: {yielded} ({yielded / n:.1%})")
    print(f"  within +/-{CLOSE_CALL_MARGIN} of runner-up: {close} ({close / n:.1%})")
    print(f"[audit-stats] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
