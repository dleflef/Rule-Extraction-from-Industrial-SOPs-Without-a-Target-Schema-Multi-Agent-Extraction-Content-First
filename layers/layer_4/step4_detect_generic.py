"""step4_detect_generic.py
==================================================
Downstream test: can the SCHEMA-AGNOSTIC extraction drive anomaly detection?

The detection stack in step5_core.py was written against the schema-specified
layer-2 output (ruleId, class, critHi, ...). The multi-agent pipeline emits a
schema it discovered for itself (id, category, crit_hi, ...), so the two do not
meet. This script is the adapter, and it is deliberately the ONLY new code:
every detector, the plausibility guard, the alarm merge, the ground-truth
scoring and the metrics are imported unchanged from step5_core, so a result
here is a statement about the extraction, not about a reimplemented detector.

No Neo4j. step5_core's detection entry points take plain Rule objects and a CSV
path; only its rule LOADER used the database.

Three translation steps, in decreasing order of defensibility:

1. Sensor resolution -- SCHEMA-AGNOSTIC. Records name sensors by the short code
   the document prints ("TMP"); the telemetry uses facility ids
   ("ST01_FILLING_TMP"). The station is located in the record BY VALUE (any
   cell holding one of the nine Component ids the seed graph declares), never
   by column name, then station + code is looked up in the declared sensor
   inventory. Nothing is invented: a record whose sensor cannot be resolved
   against the facility's own inventory is reported unresolved and dropped.

2. Field renaming -- MECHANICAL. crit_hi -> critHi and friends, by the same
   case-folding/separator-stripping normalisation step3_field_verification
   uses. No vocabulary consulted.

3. Rule class -- STRUCTURAL. The detector stack dispatches on four fixed
   classes while the pipeline discovers category labels of its own, so the
   class is decided by what a record CONTAINS rather than by what it is
   called: a record carrying numeric bounds for a resolvable sensor is a
   threshold rule, and a record carrying none falls through to the free-text
   stuck/drift/sustained parser. CATEGORY_TO_CLASS survives only to exclude
   access-control records, which no telemetry detector can consume.

   This was not the first design, and the failure is worth recording. Keying
   on the discovered label cost every threshold on the two runs that chose
   "AlarmThreshold" over "SensorThreshold": 27 detector-ready rules became 5,
   and detection F1 fell from 0.897 to 0.316 -- while F1_content reported
   0.719 +/- 0.000 on all five runs. A label whitelist over a discovered
   schema is the same instrument that failed in step3_field_verification, and
   it fails the same way.

Null control: --null-control derations rule->sensor assignment, keeping the
bounds and the rule population intact. Detection must collapse under it or the
result is a property of the telemetry rather than of the extraction. It does:
F1 0.897 -> 0.083, with 19 of 27 rules quarantined as implausible.

Usage:
    python3 step4_detect_generic.py --rules <extraction.csv>
    python3 step4_detect_generic.py --all-dev-runs
    python3 step4_detect_generic.py --all-dev-runs --null-control
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step5_core import (  # noqa: E402
    Rule,
    compute_anomaly_metrics,
    compute_violation_rates,
    load_gt_windows,
    merge_alarms,
    score_coverage,
    apply_strictness,
    stream_and_detect,
    wilson_ci,
    ON_DELAY_WARNING_MIN,
    PLAUSIBILITY_MAX_VIOLATION_RATE,
)

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NODES_CSV = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "nodes.csv")
PRED_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "detection_generic_results")

BOUND_MAP = {"crithi": "crit_hi", "warnhi": "warn_hi",
             "warnlo": "warn_lo", "critlo": "crit_lo"}

# Step 3 of the translation. One line per discovered label; everything absent
# from this table is counted as unmapped and reported.
CATEGORY_TO_CLASS = {
    # numeric operating envelopes -> threshold detection
    "sensorthreshold": "ThresholdRule",
    "standinglimit": "OperationalRule",
    "environmentalmonitoringresponse": "OperationalRule",
    "monitoringrequirement": "OperationalRule",
    # free-text conditions that step5_core parses into stuck/drift/sustained
    "predictivemaintenancetrigger": "MaintenanceRule",
    "recurringduty": "MaintenanceRule",
    "anomalydefinition": "MaintenanceRule",
    "faultdetection": "MaintenanceRule",
    # explicitly out of scope for a telemetry detector
    "accessauthorization": "AccessRule",
    "accessrule": "AccessRule",
    "occupancylimit": "AccessRule",
}

# Free-text columns concatenated into Rule.condition, which is all the
# maintenance parser reads. Order is stable so the string is reproducible.
CONDITION_FIELDS = ("rule", "condition", "trigger", "detection_criterion",
                    "definition", "requirement", "primary_criterion",
                    "pattern", "critical_response", "action")


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def to_float(v):
    try:
        f = float(str(v).strip())
        return f
    except (TypeError, ValueError):
        return None


def load_inventory() -> tuple[set, dict]:
    """The facility's declared stations and sensors. This is the seed
    knowledge graph's ABox, not the rule annotation: it describes what
    equipment exists, which any consumer of extracted rules necessarily has.

    Exactly two columns are read, `label` and `name`, and the guarantee is about
    that rather than about the file. nodes.csv does carry bound columns on its
    Sensor rows and the 14 AnomalyEvent nodes; neither is ever touched here, so
    no threshold and no episode label enters detection by this path."""
    nodes = pd.read_csv(NODES_CSV, dtype=str).fillna("")
    stations = {s.strip() for s in nodes.loc[nodes.label == "Component", "name"]
                if s.strip()}
    sensors = {s.strip() for s in nodes.loc[nodes.label == "Sensor", "name"]
               if s.strip()}
    return stations, sensors


def resolve_station(row: dict, stations: set) -> str:
    """By value, over every cell of the record — never by column name."""
    for v in row.values():
        s = str(v).strip()
        if s in stations:
            return s
    return ""


def resolve_sensor(row: dict, colmap: dict, stations: set, sensors: set) -> str:
    """Full facility sensor id, or "" when the record names no sensor the
    facility declares."""
    raw = str(row.get(colmap.get("sensor", ""), "")).strip()
    if not raw:
        return ""
    if raw in sensors:                      # already a full id
        return raw
    station = resolve_station(row, stations)
    if station and f"{station}_{raw}" in sensors:
        return f"{station}_{raw}"
    return ""


def build_rules(path: str, stations: set, sensors: set,
                dispatch: str = "structure") -> tuple[list, dict]:
    df = pd.read_csv(path, dtype=str).fillna("")
    colmap = {norm(c): c for c in df.columns}
    id_col = colmap.get("id") or colmap.get("ruleid") or df.columns[0]
    cat_col = colmap.get("category") or colmap.get("class")

    rules, stats = [], {
        "records": len(df), "unmapped_categories": {}, "unresolved_sensor": 0,
        "access_skipped": 0, "no_sensor_needed": 0,
    }
    for _, r in df.iterrows():
        row = r.to_dict()
        raw_cat = norm(row.get(cat_col, "")) if cat_col else ""
        if CATEGORY_TO_CLASS.get(raw_cat) == "AccessRule":
            stats["access_skipped"] += 1
            continue

        sensor = resolve_sensor(row, colmap, stations, sensors)
        if not sensor:
            stats["unresolved_sensor"] += 1
            continue

        bounds = {}
        for key, attr in BOUND_MAP.items():
            col = colmap.get(key)
            bounds[attr] = to_float(row.get(col, "")) if col else None

        # Dispatch STRUCTURALLY (default): a record carrying numeric bounds for
        # a resolvable sensor is a threshold rule whatever the inducer decided
        # to call it, and only records with no bounds fall through to the
        # free-text (stuck/drift/sustained) parser.
        #
        # dispatch="label" reproduces the alternative the thesis reports as
        # unsafe -- the class read off the discovered category name. It exists
        # so that the reported cost of that choice can be regenerated rather
        # than taken on trust; it is not the operating configuration.
        if dispatch == "label":
            cls = CATEGORY_TO_CLASS.get(raw_cat)
            if cls is None:
                stats["unmapped_categories"][row.get(cat_col, "")] = \
                    stats["unmapped_categories"].get(row.get(cat_col, ""), 0) + 1
                continue
        elif any(v is not None for v in bounds.values()):
            cls = "ThresholdRule"
        else:
            cls = CATEGORY_TO_CLASS.get(raw_cat, "MaintenanceRule")
            if cls == "ThresholdRule":       # labelled a threshold, carries none
                cls = "MaintenanceRule"
        stats.setdefault("dispatched", {})
        stats["dispatched"][cls] = stats["dispatched"].get(cls, 0) + 1

        condition = " ".join(
            str(row.get(colmap[f], "")).strip()
            for f in CONDITION_FIELDS if f in colmap and str(row.get(colmap[f], "")).strip())

        rules.append(Rule(
            rule_id=str(row.get(id_col, "")).strip(),
            cls=cls,
            sensor=sensor,
            crit_hi=bounds["crit_hi"], warn_hi=bounds["warn_hi"],
            warn_lo=bounds["warn_lo"], crit_lo=bounds["crit_lo"],
            condition=condition,
            action=str(row.get(colmap.get("action", ""), "")).strip(),
            source=str(row.get(colmap.get("sourcefile", ""), "")).strip(),
            severity="",
            station=resolve_station(row, stations),
        ))
    return rules, stats


def permute_sensors(rules: list, seed: int = 42) -> list:
    """Null control. Detection is only evidence about the EXTRACTION if it
    collapses when the extracted bounds are attached to the wrong sensors.
    Each rule keeps its own bounds and condition text; only the sensor it
    governs is deranged, so the rule population, the detector stack and the
    scoring are all untouched. A run that still finds the anomalies under this
    permutation is finding them from the telemetry, not from the rules."""
    import random
    rng = random.Random(seed)
    targets = [r.sensor for r in rules]
    for _ in range(100):                     # derange: no rule keeps its sensor
        rng.shuffle(targets)
        if all(t != r.sensor for t, r in zip(targets, rules)):
            break
    out = []
    for r, t in zip(rules, targets):
        out.append(Rule(rule_id=r.rule_id + "-NULL", cls=r.cls, sensor=t,
                        crit_hi=r.crit_hi, warn_hi=r.warn_hi,
                        warn_lo=r.warn_lo, crit_lo=r.crit_lo,
                        condition=r.condition, action=r.action,
                        source=r.source, severity=r.severity, station=r.station))
    return out


def strictness_report(rules: list, quarantined: set, coverage: list,
                      verbose: bool = True) -> dict:
    """Recall under stricter acceptance, at two alarm on-delays.

    Two things depress the headline figure and neither is visible in it. The
    first is that any temporal overlap counts as a detection, so a single alarm
    grazing an episode scores the same as covering it. The second is the
    warning-tier on-delay: an alarm is withheld until the condition has
    persisted, which is sound operational practice against transients but
    removes the opening minutes of every episode from the covered fraction.

    Reporting one number conflates the two. The operating point (the shipped
    on-delay) is what a plant would actually see; the zero-delay column removes
    the operational filter and is the fairer view of what the RULES support.
    Both are reported because neither alone is honest."""
    out = {}
    for tag, od in (("op", ON_DELAY_WARNING_MIN), ("nodelay", 0)):
        cov = coverage
        if od != ON_DELAY_WARNING_MIN:
            alarms, _ = stream_and_detect(rules, quarantined,
                                          on_delay_warning_min=od)
            cov = score_coverage(merge_alarms(alarms), load_gt_windows())
        # Any-overlap recall AT THIS on-delay, recorded for every on-delay
        # rather than for the operating point alone. Writing out only the
        # operating point would leave the other cells of the reported table
        # without an artifact behind them; recording each one costs nothing and
        # makes the whole table auditable from this file.
        n_any = sum(1 for c in cov if c["status"] == "COVERED")
        out[f"{tag}_recall_any"] = round(n_any / len(cov), 3) if cov else 0.0
        for lbl, kw in (("cov25", {"min_cov_pct": 25}),
                        ("cov50", {"min_cov_pct": 50}),
                        ("lat30", {"max_latency_min": 30})):
            n, _ = apply_strictness(cov, **kw)
            out[f"{tag}_recall_{lbl}"] = round(n / len(cov), 3) if cov else 0.0
        covered = [float(c["coveragePct"]) for c in cov if c["status"] == "COVERED"]
        out[f"{tag}_mean_coverage_pct"] = round(
            sum(covered) / len(covered), 1) if covered else 0.0
    if verbose:
        print(f"  strictness @ on-delay {ON_DELAY_WARNING_MIN} min (operating): "
              + "  ".join(f"R@{k.split('_')[-1]}={v}" for k, v in out.items()
                          if k.startswith("op_recall")))
        print(f"  strictness @ on-delay 0 min (rules only):      "
              + "  ".join(f"R@{k.split('_')[-1]}={v}" for k, v in out.items()
                          if k.startswith("nodelay_recall")))
        print(f"  mean episode coverage: {out['op_mean_coverage_pct']}% "
              f"(operating) vs {out['nodelay_mean_coverage_pct']}% (no delay)")
    return out


def detect(path: str, verbose: bool = True, null_control: bool = False,
           dispatch: str = "structure") -> dict:
    stations, sensors = load_inventory()
    rules, stats = build_rules(path, stations, sensors, dispatch)
    if null_control:
        rules = permute_sensors(rules)

    if verbose:
        print(f"\n=== {os.path.basename(path)} ===")
        print(f"  {stats['records']} records -> {len(rules)} detector-ready rules")
        print(f"    access-class skipped : {stats['access_skipped']}")
        print(f"    sensor unresolved    : {stats['unresolved_sensor']}")
        print(f"    dispatched           : {stats.get('dispatched', {})}")
        if stats["unmapped_categories"]:
            n = sum(stats["unmapped_categories"].values())
            print(f"    label-dispatch drops : {n} records over "
                  f"{len(stats['unmapped_categories'])} unmapped labels")
    if not rules:
        return {"file": os.path.basename(path), "rules": 0}

    vrates = compute_violation_rates(rules)
    quarantined = {rid for rid, rate in vrates.items()
                   if rate > PLAUSIBILITY_MAX_VIOLATION_RATE}
    alarms, n_rows = stream_and_detect(rules, quarantined)
    events = merge_alarms(alarms)
    coverage = score_coverage(events, load_gt_windows())
    m = compute_anomaly_metrics(coverage, events)

    noncorr = [v for v in coverage if v["type"] != "CORRELATED"]
    nc_tp = sum(1 for v in noncorr if v["status"] == "COVERED")
    nc_lo, nc_hi = wilson_ci(nc_tp, len(noncorr))

    if verbose:
        print(f"  quarantined by plausibility guard: {len(quarantined)}")
        print(f"  {n_rows:,} readings -> {len(alarms):,} alarms -> {len(events)} events")
        print(f"  GT anomalies {m['anomaly_tp']}/{m['anomaly_events_total']} covered "
              f"| recall {m['anomaly_recall']:.3f} precision {m['anomaly_precision']:.3f} "
              f"F1 {m['anomaly_f1']:.3f}")
        print(f"  excluding CORRELATED (no fusion detector): "
              f"{nc_tp}/{len(noncorr)} = {nc_tp / len(noncorr):.3f} "
              f"[{nc_lo:.3f}, {nc_hi:.3f}]")
        missed = [v["gtId"] for v in coverage if v["status"] != "COVERED"]
        if missed:
            print(f"  missed: {missed}")

    strict = strictness_report(rules, quarantined, coverage, verbose)

    return {
        "file": os.path.basename(path),
        "records": stats["records"],
        "rules_detector_ready": len(rules),
        "unmapped_records": 0,
        "unresolved_sensor": stats["unresolved_sensor"],
        "quarantined": len(quarantined),
        "events_detected": len(events),
        "gt_total": m["anomaly_events_total"],
        "gt_covered": m["anomaly_tp"],
        "recall": m["anomaly_recall"],
        "precision": m["anomaly_precision"],
        "f1": m["anomaly_f1"],
        "recall_excl_correlated": round(nc_tp / len(noncorr), 3) if noncorr else 0.0,
        **strict,
        "missed": "|".join(v["gtId"] for v in coverage if v["status"] != "COVERED"),
    }, coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", help="one extraction CSV")
    ap.add_argument("--all-dev-runs", action="store_true",
                    help="every dev-corpus run of the schema-agnostic pipeline")
    ap.add_argument("--null-control", action="store_true",
                    help="derange rule->sensor assignment; detection should collapse")
    ap.add_argument("--dispatch", choices=("structure", "label"), default="structure",
                    help="how the rule class is decided; 'label' reproduces the "
                         "unsafe alternative reported in the thesis")
    ap.add_argument("--glob", help="shell glob selecting several extraction CSVs")
    ap.add_argument("--out-suffix", default=None,
                    help="write to detection_summary<SUFFIX>.csv instead of the "
                         "headline artefact. Required when scoring anything other "
                         "than the schema-agnostic pipeline, so that a reader who "
                         "opens detection_summary.csv never has to wonder which "
                         "extraction produced it.")
    args = ap.parse_args()

    if args.glob:
        import glob
        paths = sorted(glob.glob(args.glob))
        if not paths:
            ap.error(f"--glob matched nothing: {args.glob}")
    elif args.all_dev_runs:
        import glob
        paths = sorted(glob.glob(os.path.join(
            PRED_DIR, "ext_multi_agent_generic_dev_production_line_run*_*.csv")))
    elif args.rules:
        paths = [args.rules]
    else:
        ap.error("pass --rules <csv>, --glob <pattern>, or --all-dev-runs")

    os.makedirs(OUT_DIR, exist_ok=True)
    summaries, all_cov = [], []
    for p in paths:
        out = detect(p, null_control=args.null_control, dispatch=args.dispatch)
        if isinstance(out, tuple):
            summary, coverage = out
            summaries.append(summary)
            for c in coverage:
                c["run_file"] = os.path.basename(p)
                all_cov.append(c)

    if summaries:
        cols = list(summaries[0].keys())
        # The null control and the label-dispatch reproduction are separate
        # experiments and must not overwrite the headline artefact: a reader
        # who finds detection_summary.csv should never have to wonder which
        # condition produced it.
        suffix = (args.out_suffix if args.out_suffix is not None else
                  "_null" if args.null_control else
                  "_labeldispatch" if args.dispatch == "label" else "")
        with open(os.path.join(OUT_DIR, f"detection_summary{suffix}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summaries)
        if all_cov:
            cov_cols = list(all_cov[0].keys())
            with open(os.path.join(OUT_DIR, f"gt_coverage{suffix}.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cov_cols)
                w.writeheader()
                w.writerows(all_cov)
        print(f"\n[detect-generic] wrote {OUT_DIR}/detection_summary{suffix}.csv")


if __name__ == "__main__":
    main()
