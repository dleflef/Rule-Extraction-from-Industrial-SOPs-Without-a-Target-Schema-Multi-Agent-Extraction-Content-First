"""step3_field_verification.py
==================================================
Field-level verification of numeric bound placement on the development corpus,
outside the F1_content metric.

F1_content deliberately ignores field identity: after blob flattening it can
tell that 6.5 appears in a record, not that 6.5 sits in the record's low bound.
A downstream consumer needs the stronger property, so it is checked here
directly, on the one corpus where a field correspondence exists without
inventing one: normalising both sides' column names (case folding plus
separator removal) maps the ground truth's critLo/warnLo/warnHi/critHi onto
the pipeline's crit_lo/warn_lo/warn_hi/crit_hi. No vocabulary is consulted;
the match is a string operation.

Records are keyed by station and sensor. The station is resolved PER ROW as
the first non-empty value among candidate columns (station, then asset),
because the induced schema does not name that column identically across runs.
The ground truth's sensor identifiers embed the station prefix
(ST01_FILLING_TMP), while the pipeline copies the document's own short code
(TMP); the prefix is stripped for the comparison.

For every ThresholdRule ground-truth row, each of its non-empty bound cells is
compared numerically against the same-named cell of the record with the same
station and sensor. Two artifacts are written:

    step3_results/field_verification_dev.csv             per-run totals
    step3_results/field_verification_dev_mismatches.csv  every disagreeing cell

Usage:
    python3 step3_field_verification.py
"""

import glob
import os
import re
import sys

import pandas as pd

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRED_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
OUT_DIR = os.path.join(_PROJECT_ROOT, "layers", "step3_results")
GT_PATH = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "ground_truth.csv")

BOUND_FIELDS = ("critlo", "warnlo", "warnhi", "crithi")
STATION_CANDIDATES = ("station", "asset")


def norm_col(name: str) -> str:
    """Case folding plus separator removal: critLo, crit_lo, CRIT-LO -> critlo."""
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def values_equal(gt_val: str, pred_val: str) -> bool:
    """Numeric comparison where both sides parse (30.0 == 30), else exact text."""
    g, p = str(gt_val).strip(), str(pred_val).strip()
    try:
        return float(g) == float(p)
    except ValueError:
        return g == p


def load_gt_threshold_rows() -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH, dtype=str).fillna("")
    rows = gt[(gt["class"] == "ThresholdRule") & (gt["sensor"] != "")].copy()
    rows["short_sensor"] = [
        s[len(st) + 1:] if s.startswith(st + "_") else s
        for st, s in zip(rows["station"], rows["sensor"])
    ]
    return rows


def pred_key(row: pd.Series, colmap: dict) -> tuple:
    station = ""
    for cand in STATION_CANDIDATES:
        col = colmap.get(cand)
        if col and str(row[col]).strip():
            station = str(row[col]).strip()
            break
    sensor_col = colmap.get("sensor")
    sensor = str(row[sensor_col]).strip() if sensor_col else ""
    return station, sensor


def verify_run(run_no: int, path: str, gt_rows: pd.DataFrame):
    pred = pd.read_csv(path, dtype=str).fillna("")
    colmap = {norm_col(c): c for c in pred.columns}
    bound_cols = [colmap[f] for f in BOUND_FIELDS if f in colmap]

    keyed = {}
    for _, row in pred.iterrows():
        if not any(str(row[c]).strip() for c in bound_cols):
            continue
        keyed.setdefault(pred_key(row, colmap), row)

    cells = matches = 0
    sensors_found = 0
    mismatches = []
    for _, g in gt_rows.iterrows():
        rec = keyed.get((g["station"], g["short_sensor"]))
        if rec is not None:
            sensors_found += 1
        for gt_field in ("critLo", "warnLo", "warnHi", "critHi"):
            gt_val = str(g[gt_field]).strip()
            if not gt_val:
                continue
            cells += 1
            pred_col = colmap.get(norm_col(gt_field))
            pred_val = str(rec[pred_col]).strip() if rec is not None and pred_col else ""
            if rec is not None and values_equal(gt_val, pred_val):
                matches += 1
            else:
                mismatches.append({
                    "run": run_no,
                    "station": g["station"],
                    "sensor": g["sensor"],
                    "field": gt_field,
                    "gt_value": gt_val,
                    "pred_value": pred_val if rec is not None else "(sensor not found)",
                })
    return {
        "run": run_no,
        "prediction_file": os.path.basename(path),
        "gt_threshold_sensors": len(gt_rows),
        "sensors_recovered": sensors_found,
        "bound_cells": cells,
        "cells_matching_gt": matches,
        "cells_mismatching_gt": cells - matches,
    }, mismatches


def main() -> None:
    paths = sorted(glob.glob(os.path.join(
        PRED_DIR, "ext_multi_agent_generic_dev_production_line_run*_*.csv")))
    if not paths:
        sys.exit(f"no dev prediction files under {PRED_DIR}")

    gt_rows = load_gt_threshold_rows()
    summaries, all_mismatches = [], []
    for i, path in enumerate(paths, start=1):
        summary, mism = verify_run(i, path, gt_rows)
        summaries.append(summary)
        all_mismatches.extend(mism)

    total_cells = sum(s["bound_cells"] for s in summaries)
    total_match = sum(s["cells_matching_gt"] for s in summaries)
    summaries.append({
        "run": "TOTAL",
        "prediction_file": f"{len(paths)} runs",
        "gt_threshold_sensors": len(gt_rows),
        "sensors_recovered": min(s["sensors_recovered"] for s in summaries),
        "bound_cells": total_cells,
        "cells_matching_gt": total_match,
        "cells_mismatching_gt": total_cells - total_match,
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_summary = os.path.join(OUT_DIR, "field_verification_dev.csv")
    out_mism = os.path.join(OUT_DIR, "field_verification_dev_mismatches.csv")
    pd.DataFrame(summaries).to_csv(out_summary, index=False)
    pd.DataFrame(all_mismatches).to_csv(out_mism, index=False)

    pct = 100.0 * total_match / total_cells if total_cells else 0.0
    print(f"[field-verify] {total_match}/{total_cells} bound cells "
          f"({pct:.1f}%) match the ground truth across {len(paths)} runs")
    for s in summaries[:-1]:
        print(f"  run {s['run']}: {s['cells_matching_gt']}/{s['bound_cells']} cells, "
              f"{s['sensors_recovered']}/{s['gt_threshold_sensors']} sensors recovered")
    distinct = {(m["sensor"], m["field"]) for m in all_mismatches}
    print(f"  distinct mismatching cells: {len(distinct)}")
    for sensor, field in sorted(distinct):
        print(f"    {sensor}.{field}")
    print(f"[field-verify] wrote {out_summary}")
    print(f"[field-verify] wrote {out_mism}")


if __name__ == "__main__":
    main()
