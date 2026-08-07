"""
step5_sensitivity.py

The robustness companion to step5_detect.py. Three questions are answered,
each in its own part, and each is written to its own CSV so the evidence
can be cited independently.

Part 1 — parameter sensitivity (sensitivity_analysis.csv).
Every free constant of the detection engine (warning on-delay, alarm merge
window, plausibility cutoff, drift baseline ratio, drift minimum-evidence
guard) is varied one factor at a time while everything else is held at the
shipped default. If the headline result were an artifact of tuning, it
would move here; a flat recall column is the evidence that it is not.

Part 2 — scoring strictness (scoring_strictness.csv).
The shipped scoring rule counts ANY nonzero overlap with a ground-truth
window as COVERED. In this part the already-computed baseline run is
re-scored under progressively stricter acceptance criteria: a minimum
overlap percentage, a maximum detection latency in minutes, and a maximum
latency relative to each event's own duration. The resulting curve shows
exactly how much of the headline recall is bought by the lenient rule.

Part 3 — calibration-filter equivalence (calibration_filter_check.csv).
The shipped plausibility filter is transductive: violation rates are
computed over the full evaluation stream, which a live deployment could
not do. Here the rates are recomputed over only the pre-registered first
CALIBRATION_HOURS of the stream — a commissioning period that would be
available in a real deployment — and the resulting quarantine set is
compared with the shipped one. If the two sets are identical, the
transductivity concern is reduced to an implementation detail.

The script is read-only: Neo4j is queried for the rules but never written.
It is run after step4_populate.py , step4b_load_abox.py.

Usage:
    python layers/layer_4/step5_sensitivity.py
"""

from __future__ import annotations

import csv
import os

from step5_core import (
    RESULTS_DIR,
    ON_DELAY_WARNING_MIN, GAP_MERGE_MIN, PLAUSIBILITY_MAX_VIOLATION_RATE,
    BASELINE_WINDOW_RATIO, DRIFT_MIN_REF_SAMPLES, CALIBRATION_HOURS,
    load_rules_from_neo4j, compute_violation_rates, stream_and_detect,
    merge_alarms, load_gt_windows, score_coverage, compute_anomaly_metrics,
    apply_strictness,
)

# Every config below is the shipped default with exactly one entry changed,
# so any difference in the results can be attributed to that one factor.
# A cal_window of None means the violation rates are taken over the full
# stream, as shipped; a number selects the commissioning-window variant.
_D = {"on_delay": ON_DELAY_WARNING_MIN, "merge": GAP_MERGE_MIN,
      "plausibility": PLAUSIBILITY_MAX_VIOLATION_RATE,
      "drift_ratio": BASELINE_WINDOW_RATIO,
      "drift_min_ref": DRIFT_MIN_REF_SAMPLES,
      "cal_window": None}

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
    # Each config is run through the full detect , merge , score path.
    # The first config is the shipped default; its coverage rows are kept
    # aside because Part 2 re-scores exactly that run.
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
    # No new detection is performed here: the shipped-default coverage rows
    # from Part 1 are re-scored under stricter acceptance criteria, so the
    # curve isolates the effect of the scoring rule itself.
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
    # The two violation-rate tables computed at the top of main() are
    # compared here: one taken over the full stream (as shipped) and one
    # over only the first CALIBRATION_HOURS. If both quarantine the same
    # rules, the shipped filter could have been deployed online unchanged.
    print(f"\nPart 3 — calibration-filter equivalence "
          f"(full stream vs first {CALIBRATION_HOURS} h)\n")
    cutoff = PLAUSIBILITY_MAX_VIOLATION_RATE

    def _qset(vr):
        return {rid for rid, rate in vr.items() if rate > cutoff}

    q_full, q_cal = _qset(vrates), _qset(vrates_cal)
    max_diff = max((abs(vrates[r] - vrates_cal.get(r, 0.0))
                    for r in vrates), default=0.0)
    same = q_full == q_cal
    c_rows = [{
        "stream": "original",
        "quarantine_full_stream": "|".join(sorted(q_full)),
        "quarantine_calibration_window": "|".join(sorted(q_cal)),
        "quarantine_sets_identical": str(same),
        "max_abs_rate_difference": round(max_diff, 4),
    }]
    print(f"  original: quarantine sets "
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
    if same:
        print("   An online system calibrated in the first "
              f"{CALIBRATION_HOURS} h would quarantine the same rules and "
              "produce identical results:")
        print("    the filter does not exploit anomaly-period data.")


if __name__ == "__main__":
    main()
