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

Records are keyed by station and sensor. The station is resolved PER ROW BY
VALUE, not by column name: the row is scanned for any cell holding one of the
station identifiers the seed knowledge graph declares (the nine Component
nodes of kg_seed/nodes.csv, a facility inventory that any consumer of these
records legitimately has). A column-name whitelist was tried first and is the
wrong instrument -- the induced schema does not name that column identically
across runs, so a whitelist silently reports "no station" for a record that
carries one under a name the list happens not to hold. Resolving by value is
schema-agnostic: it finds the station wherever the inducer filed it, and finds
nothing when the record genuinely carries none.

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
ABOX_NODES = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "nodes.csv")


def load_declared_stations() -> set:
    """The station identifiers the seed knowledge graph declares (Component
    nodes). Used to locate the station in a predicted record by value rather
    than by column name -- see the module docstring."""
    nodes = pd.read_csv(ABOX_NODES, dtype=str).fillna("")
    return {s.strip() for s in nodes.loc[nodes["label"] == "Component", "name"]
            if s.strip()}


def load_declared_bounds() -> dict:
    """The bounds the FACILITY declares on each sensor, read from the Sensor
    nodes of the seed knowledge graph.

    This is a second reference for the same cells the annotation covers. The
    two can disagree, and where they do it matters which one the extraction
    followed: a value that matches the plant and not the annotation is an
    annotation error, while the reverse would be an extraction error. Reporting
    only agreement with the annotation cannot tell those apart, so both counts
    are produced here and neither is corrected against the other."""
    nodes = pd.read_csv(ABOX_NODES, dtype=str).fillna("")
    sensors = nodes[nodes["label"] == "Sensor"]
    declared = {}
    for _, n in sensors.iterrows():
        name = str(n["name"]).strip()
        if not name:
            continue
        declared[name] = {f: str(n[f]).strip() for f in
                          ("critLo", "warnLo", "warnHi", "critHi") if f in sensors.columns}
    return declared


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


def pred_key(row: pd.Series, colmap: dict, stations: set) -> tuple:
    """Station by value (any column holding a declared station id), sensor by
    the record's own `sensor` field. Returns ("", sensor) when no cell of the
    row names a station the facility declares."""
    station = ""
    for val in row:
        v = str(val).strip()
        if v in stations:
            station = v
            break
    sensor_col = colmap.get("sensor")
    sensor = str(row[sensor_col]).strip() if sensor_col else ""
    return station, sensor


def verify_run(run_no: int, path: str, gt_rows: pd.DataFrame, stations: set,
               plant: dict):
    pred = pd.read_csv(path, dtype=str).fillna("")
    colmap = {norm_col(c): c for c in pred.columns}
    bound_cols = [colmap[f] for f in BOUND_FIELDS if f in colmap]

    keyed = {}
    for _, row in pred.iterrows():
        if not any(str(row[c]).strip() for c in bound_cols):
            continue
        keyed.setdefault(pred_key(row, colmap, stations), row)

    cells = matches = 0
    plant_cells = plant_matches = 0
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
            # The same cell, scored against the plant's own declared value.
            # Counted only where the facility declares one, so the denominator
            # says how many cells the comparison was actually able to make.
            plant_val = plant.get(str(g["sensor"]).strip(), {}).get(gt_field, "")
            if plant_val:
                plant_cells += 1
                if rec is not None and values_equal(plant_val, pred_val):
                    plant_matches += 1
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
                    "plant_value": plant_val or "(not declared)",
                })
    return {
        "run": run_no,
        "prediction_file": os.path.basename(path),
        "gt_threshold_sensors": len(gt_rows),
        "sensors_recovered": sensors_found,
        "bound_cells": cells,
        "cells_matching_gt": matches,
        "cells_mismatching_gt": cells - matches,
        "cells_compared_to_plant": plant_cells,
        "cells_matching_plant": plant_matches,
        "cells_mismatching_plant": plant_cells - plant_matches,
    }, mismatches


def main() -> None:
    paths = sorted(glob.glob(os.path.join(
        PRED_DIR, "ext_multi_agent_generic_dev_production_line_run*_*.csv")))
    if not paths:
        sys.exit(f"no dev prediction files under {PRED_DIR}")

    gt_rows = load_gt_threshold_rows()
    stations = load_declared_stations()
    plant = load_declared_bounds()
    summaries, all_mismatches = [], []
    for i, path in enumerate(paths, start=1):
        summary, mism = verify_run(i, path, gt_rows, stations, plant)
        summaries.append(summary)
        all_mismatches.extend(mism)

    total_cells = sum(s["bound_cells"] for s in summaries)
    total_match = sum(s["cells_matching_gt"] for s in summaries)
    total_plant_cells = sum(s["cells_compared_to_plant"] for s in summaries)
    total_plant_match = sum(s["cells_matching_plant"] for s in summaries)
    # Every field of the TOTAL row is a sum over runs, so the row can be read
    # with the same arithmetic as any single run. A minimum over runs would
    # print 0 for sensors_recovered whenever a single run failed to key, which
    # makes the pooled figure unreadable.
    summaries.append({
        "run": "TOTAL",
        "prediction_file": f"{len(paths)} runs",
        "gt_threshold_sensors": len(gt_rows) * len(paths),
        "sensors_recovered": sum(s["sensors_recovered"] for s in summaries),
        "bound_cells": total_cells,
        "cells_matching_gt": total_match,
        "cells_mismatching_gt": total_cells - total_match,
        "cells_compared_to_plant": total_plant_cells,
        "cells_matching_plant": total_plant_match,
        "cells_mismatching_plant": total_plant_cells - total_plant_match,
    })

    # Bound ordering, checked WITHOUT any reference. Agreement with the
    # annotation is one question; whether a record files a low bound in a high
    # slot is a narrower one, and it can be asked of the pipeline's own output
    # alone. A record carrying all four bounds must satisfy
    # crit_lo <= warn_lo <= warn_hi <= crit_hi; one with its bounds reversed
    # cannot. This is the perturbation F1_content prices at zero, so it is
    # established here instead.
    _BOUNDS = ["crit_lo", "warn_lo", "warn_hi", "crit_hi"]
    ordering = []
    for path in paths:
        pred = pd.read_csv(path, dtype=str).fillna("")
        if not all(c in pred.columns for c in _BOUNDS):
            continue
        with_any = with_all = violations = 0
        for _, row in pred.iterrows():
            vals = []
            for col in _BOUNDS:
                try:
                    vals.append(float(str(row[col]).strip()))
                except ValueError:
                    vals.append(None)
            if not any(v is not None for v in vals):
                continue
            with_any += 1
            if all(v is not None for v in vals):
                with_all += 1
                if not (vals[0] <= vals[1] <= vals[2] <= vals[3]):
                    violations += 1
        ordering.append({
            "run": len(ordering) + 1,
            "records_with_any_bound": with_any,
            "records_with_all_four": with_all,
            "ordering_violations": violations,
        })
    ordering.append({
        "run": "TOTAL",
        "records_with_any_bound": sum(o["records_with_any_bound"] for o in ordering),
        "records_with_all_four": sum(o["records_with_all_four"] for o in ordering),
        "ordering_violations": sum(o["ordering_violations"] for o in ordering),
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_summary = os.path.join(OUT_DIR, "field_verification_dev.csv")
    out_mism = os.path.join(OUT_DIR, "field_verification_dev_mismatches.csv")
    out_order = os.path.join(OUT_DIR, "field_verification_bound_ordering.csv")
    pd.DataFrame(summaries).to_csv(out_summary, index=False)
    pd.DataFrame(all_mismatches).to_csv(out_mism, index=False)
    pd.DataFrame(ordering).to_csv(out_order, index=False)
    tot = ordering[-1]
    print(f"[field-verify] bound ordering: {tot['records_with_any_bound']} records "
          f"carry bounds, {tot['records_with_all_four']} carry all four, "
          f"{tot['ordering_violations']} violate "
          f"crit_lo <= warn_lo <= warn_hi <= crit_hi")
    print(f"[field-verify] wrote {out_order}")

    pct = 100.0 * total_match / total_cells if total_cells else 0.0
    print(f"[field-verify] {total_match}/{total_cells} bound cells "
          f"({pct:.1f}%) match the ground truth across {len(paths)} runs")
    for s in summaries[:-1]:
        print(f"  run {s['run']}: {s['cells_matching_gt']}/{s['bound_cells']} cells, "
              f"{s['sensors_recovered']}/{s['gt_threshold_sensors']} sensors recovered")
    ppct = 100.0 * total_plant_match / total_plant_cells if total_plant_cells else 0.0
    print(f"[field-verify] {total_plant_match}/{total_plant_cells} of the same cells "
          f"({ppct:.1f}%) match the value the FACILITY declares for that sensor")
    distinct = {(m["sensor"], m["field"]) for m in all_mismatches}
    print(f"  distinct mismatching cells: {len(distinct)}")
    for sensor, field in sorted(distinct):
        print(f"    {sensor}.{field}")
    print(f"[field-verify] wrote {out_summary}")
    print(f"[field-verify] wrote {out_mism}")


if __name__ == "__main__":
    main()
