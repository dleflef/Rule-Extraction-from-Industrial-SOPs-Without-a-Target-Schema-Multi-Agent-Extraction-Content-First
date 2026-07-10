"""
step5_fusion_ablation.py

GT-0009 fusion evidence ablation — answers "did detecting the CORRELATED
event genuinely require the guide's three evidence sources, and which one
does what?" by removing each source and re-running detection on the
original stream.

The guide (§2.3.2) defines GT-0009 as requiring three-way fusion:
  (1) temporal co-occurrence in the timeseries
      (ST02_SEALING_CUR drift ↔ ST04_PACKAGING_SPD reduction)
  (2) the causal rule RULE-ST02-04 in SOP-001 §4.2
  (3) independent corroboration: SOP-003 §2 "Correlated Fault Resolution"
      names the ST02-CUR → ST04-SPD pattern ("conveyor speed drop via
      electrical coupling"). The current extraction MISSED this table
      (it extracted only MAINT-01..08 from SOP-003).

Conditions (each is one full detection pass over timeseries_raw.csv):

  A0  full-fusion (shipped)      link from extracted SOP-001 rules;
                                 co-occurrence gate; 3σ effect test.
  A1  evidence-1 ablated         effect test WITHOUT the co-occurrence
      (no temporal gate)         gate — measures what the timeseries
                                 evidence buys (expected: precision).
  A2a evidence-2 ablated         no correlated detector at all — measures
      (no causal rule)           whether the extracted rule is necessary.
  A2b evidence-2 weakened        co-occurrence kept, but no knowledge of
      (pair unknown)             WHICH pair/direction: any sensor's 3σ
                                 deviation while any other sensor recently
                                 alarmed — what "fusion without document
                                 knowledge" looks like.
  A3a evidence-3 enforced,       guide-faithful policy: a fusion link is
      actual extraction          activated ONLY if corroborated by a second
                                 independent document. Current extraction
                                 has no SOP-003 corroboration → link
                                 inactive. (The shipped pipeline (A0)
                                 deliberately activates on single-document
                                 evidence; this shows the strict
                                 alternative.)
  A3b evidence-3 enforced,       same strict policy, plus a COUNTERFACTUAL
      counterfactual extraction  corroborating rule constructed VERBATIM
                                 from the SOP-003 §2 text the extraction
                                 missed — shows the strict policy is
                                 satisfiable had extraction caught it.
                                 Clearly labelled; used in no other run.

Honesty notes:
  - Runs on the ORIGINAL stream only (GT-0009 is a dev event, already
    design-exposed). The held-out injection set is deliberately NOT reused
    here, preserving its one-shot validity.
  - The counterfactual rule in A3b contains only content present in
    layers/layer_1/texts/SOP_003_MaintenanceRules.txt (§2 table row). The
    document names the pattern but NOT the 90-minute lag, so no lag
    parameter is granted anywhere.

Output: detection_results/fusion_ablation.csv

Usage:
    python layers/layer_4/step5_fusion_ablation.py
"""

from __future__ import annotations

import os

from step5_core import (
    RESULTS_DIR, PLAUSIBILITY_MAX_VIOLATION_RATE,
    Rule,
    load_rules_from_csv, compute_violation_rates, stream_and_detect,
    merge_alarms, load_gt_windows, score_coverage, compute_anomaly_metrics,
    parse_correlation_links, fusion_evidence_report, write_csv,
)

# A3b counterfactual: what a correct extraction of SOP-003 §2 row 1 would
# have produced. Content is verbatim from the source document
# (SOP_003_MaintenanceRules.txt, "## 2. Correlated Fault Resolution"):
#   Pattern:          ST02_SEALING-CUR › → ST04_PACKAGING-SPD ↓
#   Primary Fault:    Sealing motor overload
#   Secondary Effect: Conveyor speed drop via electrical coupling
# No numeric lag appears in the document, so none is granted.
COUNTERFACTUAL_SOP003_RULE = Rule(
    rule_id="CF-SOP003-CORR-01",
    cls="MaintenanceRule",
    sensor="ST02_SEALING_CUR",
    crit_hi=None, warn_hi=None, warn_lo=None, crit_lo=None,
    condition=("ST02_SEALING_CUR rise correlates with ST04_PACKAGING_SPD "
               "conveyor speed drop via electrical coupling "
               "(sealing motor overload)"),
    action="Verify ST02-CUR trend; inspect sealing jaw; reset ST04 speed setpoint",
    source="SOP_003_MaintenanceRules.txt",
    severity="WARNING",
    station="ST02_SEALING",
)


def corroborated_links(rules: list[Rule]) -> bool:
    """Guide-faithful activation policy: every link must be corroborated by
    a second, independent source document (evidence 3)."""
    links = parse_correlation_links(rules)
    if not links:
        return False
    return all(e["three_way_complete"]
               for e in fusion_evidence_report(rules, links))


def run_condition(rules: list[Rule], fusion_mode: str) -> tuple[dict, dict]:
    vrates = compute_violation_rates(rules)
    quarantined = {rid for rid, rate in vrates.items()
                   if rate > PLAUSIBILITY_MAX_VIOLATION_RATE}
    alarms, _ = stream_and_detect(rules, quarantined, fusion_mode=fusion_mode)
    events = merge_alarms(alarms)
    coverage = score_coverage(events, load_gt_windows())
    m = compute_anomaly_metrics(coverage, events)
    gt9 = next(v for v in coverage if v["gtId"] == "GT-0009")
    n_corr = sum(1 for a in alarms if a.detector == "correlated")
    m["correlated_alarms"] = n_corr
    return m, gt9


def main() -> None:
    print("GT-0009 FUSION EVIDENCE ABLATION (original stream; holdout set "
          "deliberately untouched)\n")
    rules = load_rules_from_csv()

    conditions = []

    conditions.append(("A0_full_fusion_shipped",
                       "all evidence the pipeline holds (SOP-001 link + "
                       "co-occurrence gate + 3σ effect test)",
                       rules, "full"))
    conditions.append(("A1_no_temporal_cooccurrence",
                       "evidence (1) ablated: effect test ungated by cause "
                       "alarms",
                       rules, "ungated"))
    conditions.append(("A2a_no_causal_rule",
                       "evidence (2) ablated: correlated detector removed",
                       rules, "off"))
    conditions.append(("A2b_pair_unknown",
                       "evidence (2) weakened: co-occurrence without link "
                       "knowledge (all pairs, both directions)",
                       rules, "nopair"))

    # A3: strict corroboration policy (evidence 3 REQUIRED to activate)
    strict_ok_actual = corroborated_links(rules)
    conditions.append(("A3a_strict_corroboration_actual",
                       "evidence (3) enforced with ACTUAL extraction: link "
                       f"corroborated={strict_ok_actual} → fusion "
                       f"{'active' if strict_ok_actual else 'INACTIVE'}",
                       rules, "full" if strict_ok_actual else "off"))

    rules_cf = rules + [COUNTERFACTUAL_SOP003_RULE]
    strict_ok_cf = corroborated_links(rules_cf)
    conditions.append(("A3b_strict_corroboration_counterfactual",
                       "evidence (3) enforced with COUNTERFACTUAL SOP-003 §2 "
                       f"extraction: corroborated={strict_ok_cf} → fusion "
                       f"{'active' if strict_ok_cf else 'INACTIVE'}",
                       rules_cf, "full" if strict_ok_cf else "off"))

    rows = []
    print(f"  {'condition':<40} {'GT-0009':<9} {'cov%':<6} {'lat':<6} "
          f"{'recall':<8} {'prec':<7} {'fp':<4} corrAlarms")
    print(f"  {'-'*40} {'-'*9} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*4} {'-'*10}")
    for label, desc, cond_rules, mode in conditions:
        m, gt9 = run_condition(cond_rules, mode)
        rows.append({"condition": label, "description": desc,
                     "fusion_mode": mode,
                     "gt0009_status": gt9["status"],
                     "gt0009_coverage_pct": gt9["coveragePct"],
                     "gt0009_latency_min": gt9["latencyMin"],
                     **m})
        print(f"  {label:<40} {gt9['status']:<9} "
              f"{str(gt9['coveragePct']):<6} {str(gt9['latencyMin']):<6} "
              f"{m['anomaly_recall']:<8.3f} {m['anomaly_precision']:<7.3f} "
              f"{m['anomaly_fp']:<4} {m['correlated_alarms']}")

    write_csv(os.path.join(RESULTS_DIR, "fusion_ablation.csv"),
              ["condition", "description", "fusion_mode",
               "gt0009_status", "gt0009_coverage_pct", "gt0009_latency_min",
               "anomaly_events_total", "anomaly_tp", "anomaly_fn_gap",
               "anomaly_fp", "anomaly_recall", "anomaly_recall_ci95_lo",
               "anomaly_recall_ci95_hi", "anomaly_precision", "anomaly_f1",
               "correlated_alarms", "total_events_detected"],
              rows)
    print(f"\n  Written to detection_results/fusion_ablation.csv")


if __name__ == "__main__":
    main()
