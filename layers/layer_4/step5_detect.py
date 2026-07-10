"""
step5_detect.py

Phase 2 anomaly detection — main scoring run (iMAKS guide §3.1 Table 2, §3.3).

Pipeline:
  1. Load ACTIVE extracted rules from Neo4j (GOVERNS_ABOX-validated).
  2. Plausibility check: quarantine rules violating >50% of their sensor's
     readings (unsupervised; persisted to violation_rates.csv).
  3. Stream timeseries_raw.csv reading ONLY timestamp/sensor_id/value.
     Detectors: threshold, stuck, drift, sustained.
  4. Merge alarms → events; load GT windows (only now); score COVERED/GAP
     per AnomalyEvent uniformly.
  5. Maintenance node coverage (structural Neo4j check) + combined Phase 2
     fraction — labelled as mixing detection with a structural check.

GT-0009 (CORRELATED): no multi-source fusion detector is implemented (see
step5_core.py docstring). It is scored by the same uniform rule as every
other event and is expected to show as GAP; the guide's separate binary
(§3.3) is reported as NOT ATTEMPTED. The aggregate excluding it is also
reported, since the guide keeps the binary out of the aggregate.

All detector logic and metric definitions live in step5_core.py (shared with
step5_sensitivity.py, step5_baselines.py, step5_holdout.py).

Run after: step4_populate.py → step4b_load_abox.py

Usage:
    python layers/layer_4/step5_detect.py
"""

from __future__ import annotations

import os

from step5_core import (
    RESULTS_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    PLAUSIBILITY_MAX_VIOLATION_RATE, PHASE2_COVERAGE_THRESHOLD,
    load_rules_from_neo4j, compute_violation_rates, stream_and_detect,
    merge_alarms, load_gt_windows, score_coverage, compute_anomaly_metrics,
    wilson_ci, write_csv,
)


# ── Neo4j-side steps (main run only) ──────────────────────────────────────────

def evaluate_maintenance_coverage() -> list[dict]:
    """Structural check (guide Table 2, Maintenance rows): a Maintenance node
    is COVERED if at least one extracted MaintenanceRule governs its sensor.
    No detection involved — do not mix with detection scores unlabelled."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(session.run(
            "MATCH (s:ABoxNode:Sensor)-[:TRIGGERS]->(m:ABoxNode:Maintenance) "
            "OPTIONAL MATCH (r:Rule)-[:GOVERNS_ABOX]->(s) "
            "WHERE r.class = 'MaintenanceRule' "
            "RETURN m.nodeId AS nodeId, m.ruleId AS maintLabel, "
            "       s.name AS sensor, collect(r.ruleId) AS matchedRules"))
        for rec in records:
            matched = [r for r in rec["matchedRules"] if r]
            status  = "COVERED" if matched else "GAP"
            session.run(
                "MATCH (m:ABoxNode:Maintenance {nodeId: $nid}) "
                "SET m.detectionStatus = $status",
                nid=rec["nodeId"], status=status)
            for rid in matched:
                session.run(
                    "MATCH (r:Rule {ruleId: $rid}), "
                    "      (m:ABoxNode:Maintenance {nodeId: $nid}) "
                    "MERGE (r)-[:COVERS]->(m)",
                    rid=rid, nid=rec["nodeId"])
            results.append({
                "maintLabel":   rec["maintLabel"] or rec["nodeId"],
                "nodeId":       rec["nodeId"],
                "sensor":       rec["sensor"] or "",
                "status":       status,
                "matchedRules": "|".join(matched),
            })
    driver.close()
    return sorted(results, key=lambda r: r["maintLabel"])


def write_covers_to_neo4j(coverage: list[dict]) -> None:
    """detectionStatus + COVERS edges for anomaly events."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as session:
        for v in coverage:
            session.run(
                "MATCH (a:ABoxNode:AnomalyEvent {gtId: $gtId}) "
                "SET a.detectionStatus = $status",
                gtId=v["gtId"], status=v["status"])
            if v["status"] != "COVERED":
                continue
            for rid in v["matchedRules"].split("|"):
                rid = rid.strip()
                if rid:
                    session.run(
                        "MATCH (r:Rule {ruleId: $rid}), "
                        "      (a:ABoxNode:AnomalyEvent {gtId: $gtId}) "
                        "MERGE (r)-[:COVERS]->(a)",
                        rid=rid, gtId=v["gtId"])
    driver.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/6] Loading active rules from Neo4j …")
    rules = load_rules_from_neo4j()
    if not rules:
        print("  ERROR: no rules loaded. Run step4_populate.py and step4b_load_abox.py first.")
        return
    print(f"  {len(rules)} active rules (AccessRule excluded)")

    print("[2/6] Plausibility check (violation rates; no GT labels) …")
    vrates = compute_violation_rates(rules)
    quarantined = {rid for rid, rate in vrates.items()
                   if rate > PLAUSIBILITY_MAX_VIOLATION_RATE}
    print(f"  Quarantined: {sorted(quarantined) or 'none'}")
    write_csv(
        os.path.join(RESULTS_DIR, "violation_rates.csv"),
        ["ruleId", "violationRate", "quarantined", "threshold"],
        [{"ruleId": rid, "violationRate": round(rate, 4),
          "quarantined": str(rid in quarantined),
          "threshold": PLAUSIBILITY_MAX_VIOLATION_RATE}
         for rid, rate in sorted(vrates.items(), key=lambda kv: -kv[1])])

    print("[3/6] Streaming timeseries_raw.csv "
          "(reading ONLY timestamp/sensor_id/value) …")
    alarms, row_count = stream_and_detect(rules, quarantined)
    print(f"  {row_count:,} readings → {len(alarms):,} alarms")

    print("[4/6] Merging alarms into events …")
    events = merge_alarms(alarms)
    print(f"  {len(events)} detected events")

    print("[5/6] Loading GT and scoring (post-hoc) …")
    gt_windows = load_gt_windows()
    coverage   = score_coverage(events, gt_windows)
    m = compute_anomaly_metrics(coverage, events)

    # The guide (§3.3) scores GT-0009 separately from the aggregate; with no
    # fusion detector the binary is NOT ATTEMPTED, and the aggregate over
    # the remaining events is reported alongside the full one.
    noncorr    = [v for v in coverage if v["type"] != "CORRELATED"]
    nc_tp      = sum(1 for v in noncorr if v["status"] == "COVERED")
    nc_lo, nc_hi = wilson_ci(nc_tp, len(noncorr))

    def _gt_num(v: dict) -> int:
        try:
            return int(v["gtId"].rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return 0
    dev_cov  = [v for v in coverage if 1 <= _gt_num(v) <= 9]
    test_cov = [v for v in coverage if _gt_num(v) >= 10]
    dev_tp   = sum(1 for v in dev_cov  if v["status"] == "COVERED")
    test_tp  = sum(1 for v in test_cov if v["status"] == "COVERED")

    print("      Maintenance node coverage (structural Neo4j check) …")
    maint = evaluate_maintenance_coverage()
    maint_tp = sum(1 for x in maint if x["status"] == "COVERED")

    phase2_total   = len(coverage) + len(maint)
    phase2_covered = m["anomaly_tp"] + maint_tp
    phase2_frac    = phase2_covered / phase2_total if phase2_total else 0.0
    phase2_pass    = phase2_frac >= PHASE2_COVERAGE_THRESHOLD

    print("[6/6] Writing results …")
    write_csv(
        os.path.join(RESULTS_DIR, "detected_events.csv"),
        ["sensor", "start", "end", "severity", "detector", "ruleIds", "nAlarms"],
        [{"sensor": e.sensor, "start": e.start.isoformat(),
          "end": e.end.isoformat(), "severity": e.severity,
          "detector": "|".join(e.detectors), "ruleIds": "|".join(e.rule_ids),
          "nAlarms": e.n_alarms} for e in events])
    write_csv(
        os.path.join(RESULTS_DIR, "phase2_anomaly_coverage.csv"),
        ["gtId", "sensor", "type", "status", "detector", "matchedRules",
         "gtStart", "gtEnd", "detStart", "detEnd", "latencyMin", "coveragePct"],
        coverage)
    write_csv(
        os.path.join(RESULTS_DIR, "phase2_maintenance_coverage.csv"),
        ["maintLabel", "nodeId", "sensor", "status", "matchedRules"],
        maint)
    summary = [
        {"metric": "anomaly_events_total",         "value": m["anomaly_events_total"]},
        {"metric": "anomaly_tp",                   "value": m["anomaly_tp"]},
        {"metric": "anomaly_fn_gap",               "value": m["anomaly_fn_gap"]},
        {"metric": "anomaly_fp",                   "value": m["anomaly_fp"]},
        {"metric": "anomaly_recall",               "value": m["anomaly_recall"]},
        {"metric": "anomaly_recall_ci95_lo",       "value": m["anomaly_recall_ci95_lo"]},
        {"metric": "anomaly_recall_ci95_hi",       "value": m["anomaly_recall_ci95_hi"]},
        {"metric": "anomaly_precision",            "value": m["anomaly_precision"]},
        {"metric": "anomaly_f1",                   "value": m["anomaly_f1"]},
        {"metric": "anomaly_recall_excl_gt0009",   "value": round(nc_tp / len(noncorr), 3) if noncorr else 0.0},
        {"metric": "anomaly_excl_gt0009_ci95_lo",  "value": round(nc_lo, 3)},
        {"metric": "anomaly_excl_gt0009_ci95_hi",  "value": round(nc_hi, 3)},
        {"metric": "gt0009_binary",                "value": "NOT_ATTEMPTED (no fusion detector)"},
        {"metric": "anomaly_dev_total",            "value": len(dev_cov)},
        {"metric": "anomaly_dev_covered",          "value": dev_tp},
        {"metric": "anomaly_test_total",           "value": len(test_cov)},
        {"metric": "anomaly_test_covered",         "value": test_tp},
        {"metric": "maintenance_total",            "value": len(maint)},
        {"metric": "maintenance_covered",          "value": maint_tp},
        {"metric": "phase2_total_events",          "value": phase2_total},
        {"metric": "phase2_covered",               "value": phase2_covered},
        {"metric": "phase2_fraction",              "value": round(phase2_frac, 3)},
        {"metric": "phase2_threshold",             "value": PHASE2_COVERAGE_THRESHOLD},
        {"metric": "phase2_pass",                  "value": str(phase2_pass)},
        {"metric": "rules_quarantined",            "value": len(quarantined)},
        {"metric": "total_readings_streamed",      "value": row_count},
        {"metric": "total_alarms_raised",          "value": len(alarms)},
        {"metric": "total_events_detected",        "value": len(events)},
    ]
    write_csv(os.path.join(RESULTS_DIR, "phase2_summary.csv"),
              ["metric", "value"], summary)

    write_covers_to_neo4j(coverage)

    # ── Terminal summary ───────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  PHASE 2 RESULTS  (iMAKS guide Table 2)")
    print("=" * 64)
    print(f"\n  Anomaly events — all {len(coverage)} scored uniformly (any overlap):")
    print(f"    COVERED {m['anomaly_tp']}  GAP {m['anomaly_fn_gap']}  "
          f"FP {m['anomaly_fp']}")
    print(f"    Recall {m['anomaly_recall']:.3f} "
          f"(95% CI [{m['anomaly_recall_ci95_lo']:.3f}, "
          f"{m['anomaly_recall_ci95_hi']:.3f}])  "
          f"Precision {m['anomaly_precision']:.3f}  F1 {m['anomaly_f1']:.3f}")
    print(f"    Excl. GT-0009 (guide: binary scored separately): "
          f"{nc_tp}/{len(noncorr)} = {nc_tp/len(noncorr):.3f} "
          f"[{nc_lo:.3f}, {nc_hi:.3f}]")
    print(f"    Dev {dev_tp}/{len(dev_cov)}   Test {test_tp}/{len(test_cov)} "
          f"(test NOT held out during design — descriptive only, "
          f"see holdout run + EVALUATION_LIMITATIONS.md)")
    print(f"\n  {'GT ID':<10} {'Sensor':<28} {'Type':<14} {'Status':<9} "
          f"{'Cov%':<6} {'Lat(min)'}")
    for v in coverage:
        print(f"  {v['gtId']:<10} {v['sensor']:<28} {v['type']:<14} "
              f"{v['status']:<9} {str(v['coveragePct']):<6} {v['latencyMin']}")
    print(f"\n  GT-0009 binary (guide §3.3, separate from aggregate): "
          f"NOT ATTEMPTED")
    print(f"    No multi-source fusion detector is implemented; the "
          f"CORRELATED event is scored")
    print(f"    by the same uniform rule as every other event and shows as "
          f"GAP above.")
    print(f"\n  Maintenance nodes (STRUCTURAL check — no detection involved):")
    for x in maint:
        print(f"    {x['maintLabel']:<12} {x['sensor']:<28} {x['status']}")
    print(f"\n  Phase 2 combined: {phase2_covered}/{phase2_total} "
          f"({phase2_frac:.1%})  [bar ≥ {PHASE2_COVERAGE_THRESHOLD:.0%}]  "
          f"→ {'PASS' if phase2_pass else 'FAIL'}")
    print(f"    (mixes detection {m['anomaly_tp']}/{len(coverage)} with the "
          f"structural check {maint_tp}/{len(maint)} — report separately)")
    print(f"\n  Results in detection_results/: phase2_anomaly_coverage.csv, "
          f"phase2_maintenance_coverage.csv,")
    print(f"  phase2_summary.csv, detected_events.csv, violation_rates.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()
