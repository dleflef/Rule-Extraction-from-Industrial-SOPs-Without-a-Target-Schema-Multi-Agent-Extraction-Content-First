"""
step5_baselines.py

Brackets the extracted-rules Phase 2 result between an upper and lower bound
(guide §4 use-case 2 — the dataset as a standalone anomaly benchmark):

  extracted-rules    — the actual pipeline (rules rebuilt from step4_results
                       CSVs; no Neo4j needed; cross-checked vs phase2_summary).
  oracle-thresholds  — UPPER BOUND for threshold extraction: GT bounds from
                       nodes.csv through the identical machinery. The ONLY
                       config anywhere that reads GT thresholds — by design.
                       No maintenance rules, so anomalies that stay inside
                       the bounds (e.g. in-range STUCK) are structurally
                       invisible to it.
  zscore (z grid)    — LOWER BOUND: generic rolling z-score, no rules/KG/LLM.
                       Grid over z so the baseline can't be accused of being
                       tuned to lose. Same merge + scoring as the pipeline.

Outputs: detection_results/baseline_comparison.csv, baseline_per_event.csv

Usage:
    python layers/layer_4/step5_baselines.py
"""

from __future__ import annotations

import csv
import os
from collections import deque
from datetime import datetime

from step5_core import (
    RESULTS_DIR, TIMESERIES_CSV, NODES_CSV, SAMPLE_INTERVAL_SEC,
    PLAUSIBILITY_MAX_VIOLATION_RATE,
    Rule, Alarm,
    load_rules_from_csv, compute_violation_rates, stream_and_detect,
    merge_alarms, load_gt_windows, score_coverage, compute_anomaly_metrics,
    write_csv, _to_float,
)

PHASE2_SUMMARY_CSV = os.path.join(RESULTS_DIR, "phase2_summary.csv")

ZSCORE_WINDOW_SAMPLES = round(60 * 60 / SAMPLE_INTERVAL_SEC)   # 60 min
ZSCORE_WARMUP_SAMPLES = 20                                     # 10 min
ZSCORE_GRID           = [2.5, 3.0, 3.5]
ZSCORE_STD_FLOOR      = 1e-9   # frozen window → no z-score, not an alarm


def load_oracle_rules() -> list[Rule]:
    """One ThresholdRule per sensor, bounds verbatim from GT (nodes.csv) —
    the deliberate leak-everything upper bound."""
    rules = []
    with open(NODES_CSV, newline="", encoding="utf-8") as f:
        for n in csv.DictReader(f):
            if n.get("label") != "Sensor":
                continue
            b = {k: _to_float(n.get(k)) for k in
                 ("critHi", "warnHi", "warnLo", "critLo")}
            if all(v is None for v in b.values()):
                continue
            rules.append(Rule(
                rule_id=f"ORACLE-{n['name']}", cls="ThresholdRule",
                sensor=n["name"],
                crit_hi=b["critHi"], warn_hi=b["warnHi"],
                warn_lo=b["warnLo"], crit_lo=b["critLo"],
                condition="", action="", source="kg_seed/nodes.csv (GT)"))
    return rules


def stream_zscore(z_thresh: float) -> list[Alarm]:
    """Rolling z-score per sensor; window statistics computed BEFORE the
    current value is appended so a spike cannot mask itself. Reads only
    timestamp/sensor_id/value."""
    wins:   dict[str, deque] = {}
    sums:   dict[str, float] = {}
    sumsqs: dict[str, float] = {}
    alarms: list[Alarm] = []

    with open(TIMESERIES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["sensor_id"]
            try:
                val = float(row["value"])
            except (ValueError, KeyError):
                continue
            ts = datetime.fromisoformat(row["timestamp"])

            if sid not in wins:
                wins[sid], sums[sid], sumsqs[sid] = deque(), 0.0, 0.0
            win = wins[sid]

            n = len(win)
            if n >= ZSCORE_WARMUP_SAMPLES:
                mean = sums[sid] / n
                var  = max(0.0, sumsqs[sid] / n - mean * mean)
                std  = var ** 0.5
                if std > ZSCORE_STD_FLOOR and abs(val - mean) / std > z_thresh:
                    alarms.append(Alarm(sid, ts, val, "WARNING", "zscore",
                                        f"ZSCORE-{z_thresh:g}"))

            win.append(val)
            sums[sid]   += val
            sumsqs[sid] += val * val
            if len(win) > ZSCORE_WINDOW_SAMPLES:
                old = win.popleft()
                sums[sid]   -= old
                sumsqs[sid] -= old * old
    return alarms


def run_rule_config(rules: list[Rule]) -> list:
    vrates = compute_violation_rates(rules)
    quarantined = {rid for rid, rate in vrates.items()
                   if rate > PLAUSIBILITY_MAX_VIOLATION_RATE}
    alarms, _ = stream_and_detect(rules, quarantined)
    return merge_alarms(alarms)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gt_windows = load_gt_windows()
    configs = []

    print("[1/3] extracted-rules (pipeline, no Neo4j) …")
    extracted = load_rules_from_csv()
    print(f"      {len(extracted)} active extracted rules")
    configs.append(("extracted-rules",
                    "LLM-extracted rules; the actual pipeline result",
                    run_rule_config(extracted)))

    print("[2/3] oracle-thresholds (GT bounds — upper bound) …")
    oracle = load_oracle_rules()
    print(f"      {len(oracle)} oracle rules")
    configs.append(("oracle-thresholds",
                    "perfect threshold extraction; no maintenance rules "
                    "(cannot catch in-range anomalies by construction)",
                    run_rule_config(oracle)))

    print(f"[3/3] rolling z-score grid z={ZSCORE_GRID} …")
    for z in ZSCORE_GRID:
        configs.append((f"zscore-z{z:g}",
                        "no rules / no KG / no LLM",
                        merge_alarms(stream_zscore(z))))

    rows = []
    per_event = {gt.gt_id: {"gtId": gt.gt_id, "sensor": gt.sensor,
                            "type": gt.atype} for gt in gt_windows}
    print(f"\n  {'config':<20} {'tp':<6} {'recall':<8} {'ci95':<16} "
          f"{'precision':<10} {'f1':<7} fp")
    for label, notes, events in configs:
        coverage = score_coverage(events, gt_windows)
        m = compute_anomaly_metrics(coverage, events)
        rows.append({"config": label, "notes": notes, **m})
        for v in coverage:
            per_event[v["gtId"]][label] = v["status"]
        tp_str = f"{m['anomaly_tp']}/{m['anomaly_events_total']}"
        print(f"  {label:<20} {tp_str:<6} {m['anomaly_recall']:<8.3f} "
              f"[{m['anomaly_recall_ci95_lo']:.3f},{m['anomaly_recall_ci95_hi']:.3f}]  "
              f"{m['anomaly_precision']:<10.3f} {m['anomaly_f1']:<7.3f} "
              f"{m['anomaly_fp']}")

    write_csv(os.path.join(RESULTS_DIR, "baseline_comparison.csv"),
              ["config", "notes", "anomaly_events_total", "anomaly_tp",
               "anomaly_fn_gap", "anomaly_fp", "anomaly_recall",
               "anomaly_recall_ci95_lo", "anomaly_recall_ci95_hi",
               "anomaly_precision", "anomaly_f1", "total_events_detected"],
              rows)
    write_csv(os.path.join(RESULTS_DIR, "baseline_per_event.csv"),
              ["gtId", "sensor", "type"] + [c[0] for c in configs],
              [per_event[gt.gt_id] for gt in gt_windows])
    print(f"\n  Written to baseline_comparison.csv / baseline_per_event.csv")

    # Consistency check vs the Neo4j-driven main run
    if os.path.exists(PHASE2_SUMMARY_CSV):
        with open(PHASE2_SUMMARY_CSV, newline="", encoding="utf-8") as f:
            summary = {r["metric"]: r["value"] for r in csv.DictReader(f)}
        want = (summary.get("anomaly_tp"), summary.get("anomaly_fp"))
        got  = (str(rows[0]["anomaly_tp"]), str(rows[0]["anomaly_fp"]))
        if want == got:
            print(f"  ✓ extracted-rules matches phase2_summary.csv "
                  f"(tp={got[0]}, fp={got[1]}) — CSV rules ≡ Neo4j rules.")
        else:
            print(f"  ✗ MISMATCH vs phase2_summary.csv: tp/fp {got} vs {want} "
                  f"— rerun step4_populate.py → step4b → step5_detect.py.")


if __name__ == "__main__":
    main()
