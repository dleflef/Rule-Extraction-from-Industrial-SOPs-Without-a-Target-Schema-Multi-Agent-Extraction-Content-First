"""
step5_sensitivity.py

Part 1 — one-factor-at-a-time grid over every step5 free parameter
(on-delay, merge window, plausibility cutoff, drift baseline ratio, drift
minimum-evidence guard) → sensitivity_analysis.csv.

Part 2 — scoring-strictness sweep: re-scores the shipped-default run under
progressively stricter acceptance criteria (minimum overlap %, maximum
latency absolute / relative to GT duration) → scoring_strictness.csv.
The shipped rule counts ANY nonzero overlap as COVERED; this sweep is the
honest curve showing how much of the headline recall that leniency buys.

Part 3 — calibration-filter equivalence check: the shipped plausibility
filter is transductive (violation rates over the full evaluation stream).
This part computes the rates over only the pre-registered first
CALIBRATION_HOURS (a commissioning period, deployable online) and compares
the resulting quarantine set against the shipped one — on the original
stream and, if present, on the holdout injected stream. If the sets match,
the transductivity caveat reduces to an implementation detail.
→ calibration_filter_check.csv.

Read-only: never writes to Neo4j. Run after step4_populate.py →
step4b_load_abox.py.

Usage:
    python layers/layer_4/step5_sensitivity.py
"""

from __future__ import annotations

import csv
import os

from step5_core import (
    RESULTS_DIR, TIMESERIES_CSV,
    ON_DELAY_WARNING_MIN, GAP_MERGE_MIN, PLAUSIBILITY_MAX_VIOLATION_RATE,
    BASELINE_WINDOW_RATIO, DRIFT_MIN_REF_SAMPLES, CALIBRATION_HOURS,
    load_rules_from_neo4j, compute_violation_rates, stream_and_detect,
    merge_alarms, load_gt_windows, score_coverage, compute_anomaly_metrics,
    apply_strictness,
)

HOLDOUT_INJECTED_CSV = os.path.join(RESULTS_DIR, "holdout",
                                    "injected_timeseries.csv")

_D = {"on_delay": ON_DELAY_WARNING_MIN, "merge": GAP_MERGE_MIN,
      "plausibility": PLAUSIBILITY_MAX_VIOLATION_RATE,
      "drift_ratio": BASELINE_WINDOW_RATIO,
      "drift_min_ref": DRIFT_MIN_REF_SAMPLES,
      "cal_window": None}   # None = shipped full-stream violation rates

CONFIGS = [
    {**_D, "label": "baseline (shipped defaults)"},
    {**_D, "label": "on_delay=0min (no confirmation delay)", "on_delay": 0},
    {**_D, "label": "on_delay=10min (2x shipped)",           "on_delay": 10},
    {**_D, "label": "merge=5min (1/3 shipped)",              "merge": 5},
    {**_D, "label": "merge=30min (2x shipped)",              "merge": 30},
    {**_D, "label": "plausibility=off (no quarantine)",      "plausibility": 2.0},
    {**_D, "label": "plausibility=0.7 (looser cutoff)",      "plausibility": 0.7},
    {**_D, "label": "plausibility=0.3 (stricter cutoff)",    "plausibility": 0.3},
    {**_D, "label": "plausibility=calibration-8h (deployable)",
     "cal_window": CALIBRATION_HOURS},
    {**_D, "label": "drift_ratio=1 (baseline = 1x window)",  "drift_ratio": 1},
    {**_D, "label": "drift_ratio=3 (baseline = 3x window)",  "drift_ratio": 3},
    {**_D, "label": "drift_min_ref=2 (almost no guard)",     "drift_min_ref": 2},
    {**_D, "label": "drift_min_ref=40 (4x shipped)",         "drift_min_ref": 40},
]

MIN_COVERAGE_PCTS = [0, 5, 10, 25, 50, 75]
MAX_LATENCY_MINS  = [None, 120, 60, 30, 15, 5]
MAX_LATENCY_FRACS = [1.0, 0.5, 0.25]


def main() -> None:
    print("Loading rules from Neo4j (read-only) …")
    rules = load_rules_from_neo4j()
    if not rules:
        print("  ERROR: no rules loaded. Run step4_populate.py and step4b_load_abox.py first.")
        return
    vrates     = compute_violation_rates(rules)
    vrates_cal = compute_violation_rates(rules,
                                         calibration_hours=CALIBRATION_HOURS)
    gt_windows = load_gt_windows()

    # ── Part 1: parameter grid ────────────────────────────────────────────────
    print(f"\nPart 1 — parameter grid: {len(CONFIGS)} configs …\n")
    print(f"  {'config':<42} {'tp':<6} {'recall':<8} {'precision':<10} "
          f"{'f1':<7} {'fp':<4} quar")
    print(f"  {'-'*42} {'-'*6} {'-'*8} {'-'*10} {'-'*7} {'-'*4} {'-'*4}")

    rows, baseline_coverage = [], []
    for i, cfg in enumerate(CONFIGS):
        vr = vrates_cal if cfg["cal_window"] else vrates
        quarantined = {rid for rid, rate in vr.items()
                       if rate > cfg["plausibility"]}
        alarms, _ = stream_and_detect(
            rules, quarantined,
            on_delay_warning_min=cfg["on_delay"],
            baseline_window_ratio=cfg["drift_ratio"],
            drift_min_ref_samples=cfg["drift_min_ref"])
        events   = merge_alarms(alarms, gap_merge_min=cfg["merge"])
        coverage = score_coverage(events, gt_windows)
        m = compute_anomaly_metrics(coverage, events)
        if i == 0:
            baseline_coverage = coverage

        rows.append({"config": cfg["label"],
                     "on_delay_warning_min": cfg["on_delay"],
                     "gap_merge_min": cfg["merge"],
                     "plausibility_cutoff": cfg["plausibility"],
                     "calibration_window_h": cfg["cal_window"] or "",
                     "drift_baseline_ratio": cfg["drift_ratio"],
                     "drift_min_ref_samples": cfg["drift_min_ref"],
                     "rules_quarantined": len(quarantined),
                     "quarantined_rule_ids": "|".join(sorted(quarantined)),
                     **m})
        tp_str = f"{m['anomaly_tp']}/{m['anomaly_events_total']}"
        print(f"  {cfg['label']:<42} {tp_str:<6} {m['anomaly_recall']:<8.3f} "
              f"{m['anomaly_precision']:<10.3f} {m['anomaly_f1']:<7.3f} "
              f"{m['anomaly_fp']:<4} {len(quarantined)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sensitivity_analysis.csv")
    fields = ["config", "on_delay_warning_min", "gap_merge_min",
              "plausibility_cutoff", "calibration_window_h",
              "drift_baseline_ratio",
              "drift_min_ref_samples", "rules_quarantined",
              "quarantined_rule_ids", "anomaly_events_total", "anomaly_tp",
              "anomaly_fn_gap", "anomaly_fp", "anomaly_recall",
              "anomaly_recall_ci95_lo", "anomaly_recall_ci95_hi",
              "anomaly_precision", "anomaly_f1", "total_events_detected"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rec = [r["anomaly_recall"] for r in rows]
    print(f"\n  Recall range: {min(rec):.3f}–{max(rec):.3f} "
          f"(spread {max(rec)-min(rec):.3f})")
    print(f"  Written to {out_path}")

    # ── Part 2: scoring-strictness sweep ──────────────────────────────────────
    total = len(baseline_coverage)
    print(f"\nPart 2 — scoring strictness (shipped rule: ANY overlap = COVERED)\n")
    print(f"  {'criterion':<26} {'value':<20} {'tp':<7} {'recall':<8} dropped")
    s_rows = []

    def _emit(criterion, value_label, tp, dropped):
        recall = tp / total if total else 0.0
        s_rows.append({"criterion": criterion, "value": value_label,
                       "tp": tp, "total": total, "recall": round(recall, 3),
                       "dropped_gtIds": "|".join(dropped)})
        print(f"  {criterion:<26} {value_label:<20} {tp}/{total:<4} "
              f"{recall:<8.3f} {'|'.join(dropped) or '—'}")

    for pct in MIN_COVERAGE_PCTS:
        tp, dropped = apply_strictness(baseline_coverage, min_cov_pct=pct)
        _emit("min_coverage_pct",
              f">={pct}%" + (" (shipped)" if pct == 0 else ""), tp, dropped)
    for lat in MAX_LATENCY_MINS:
        tp, dropped = apply_strictness(baseline_coverage, max_latency_min=lat)
        _emit("max_latency_min",
              "unlimited (shipped)" if lat is None else f"<={lat}min",
              tp, dropped)
    for frac in MAX_LATENCY_FRACS:
        tp, dropped = apply_strictness(baseline_coverage, max_latency_frac=frac)
        _emit("max_latency_frac_of_gt", f"<={frac:g}x GT duration", tp, dropped)

    s_path = os.path.join(RESULTS_DIR, "scoring_strictness.csv")
    with open(s_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["criterion", "value", "tp", "total",
                                          "recall", "dropped_gtIds"])
        w.writeheader()
        w.writerows(s_rows)
    print(f"\n  Written to {s_path} — report this curve next to the headline recall.")

    # ── Part 3: calibration-filter equivalence check ──────────────────────────
    print(f"\nPart 3 — calibration-filter equivalence "
          f"(full stream vs first {CALIBRATION_HOURS} h)\n")
    cutoff = PLAUSIBILITY_MAX_VIOLATION_RATE

    def _qset(vr):
        return {rid for rid, rate in vr.items() if rate > cutoff}

    streams = [("original", TIMESERIES_CSV, vrates, vrates_cal)]
    if os.path.exists(HOLDOUT_INJECTED_CSV):
        streams.append((
            "holdout_injected(seed=canonical)", HOLDOUT_INJECTED_CSV,
            compute_violation_rates(rules, timeseries_csv=HOLDOUT_INJECTED_CSV),
            compute_violation_rates(rules, timeseries_csv=HOLDOUT_INJECTED_CSV,
                                    calibration_hours=CALIBRATION_HOURS)))

    c_rows = []
    for name, _path, vr_full, vr_cal in streams:
        q_full, q_cal = _qset(vr_full), _qset(vr_cal)
        max_diff = max((abs(vr_full[r] - vr_cal.get(r, 0.0))
                        for r in vr_full), default=0.0)
        same = q_full == q_cal
        c_rows.append({
            "stream": name,
            "quarantine_full_stream": "|".join(sorted(q_full)),
            "quarantine_calibration_window": "|".join(sorted(q_cal)),
            "quarantine_sets_identical": str(same),
            "max_abs_rate_difference": round(max_diff, 4),
        })
        print(f"  {name}: quarantine sets "
              f"{'IDENTICAL' if same else 'DIFFER'} "
              f"(max per-rule rate diff {max_diff:.4f})")
        if not same:
            print(f"    full-only: {sorted(q_full - q_cal)}")
            print(f"    cal-only : {sorted(q_cal - q_full)}")

    c_path = os.path.join(RESULTS_DIR, "calibration_filter_check.csv")
    with open(c_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stream", "quarantine_full_stream",
            "quarantine_calibration_window", "quarantine_sets_identical",
            "max_abs_rate_difference"])
        w.writeheader()
        w.writerows(c_rows)
    print(f"\n  Written to {c_path}")
    if all(r["quarantine_sets_identical"] == "True" for r in c_rows):
        print("  → An online system calibrated in the first "
              f"{CALIBRATION_HOURS} h would quarantine the same rules and "
              "produce identical results:")
        print("    the filter does not exploit anomaly-period data.")


if __name__ == "__main__":
    main()
