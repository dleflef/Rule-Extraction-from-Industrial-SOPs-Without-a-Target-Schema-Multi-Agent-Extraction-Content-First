"""oracle_gt_detect.py
==================================================
The oracle: what does detection achieve when the rules are PERFECT?

This experiment is deliberately isolated from the extraction pipeline. It reads
the human-written annotation directly -- the same rules a person produced by
reading the four source documents -- and drives detection with them. No
extracted record is involved, no knowledge-graph instance is touched, and the
outputs are written to this directory alone.

Why it is needed. Detection accuracy is reported against a lenient acceptance
rule (any temporal overlap with a labelled episode). Under stricter criteria
the pipeline's recall falls steeply, and that decline is uninterpretable on its
own: it could mean the extracted rules are poor, or it could mean the criteria
are unreachable for ANY rule set given these detectors and these faults. Those
two possibilities are distinguished by one measurement -- run the same
detectors, over the same telemetry, scored the same way, on rules that are
correct by construction. Whatever the oracle cannot achieve is a ceiling
imposed by the task, not a deficiency of the extractor.

The detector library is shared with the pipeline harnesses ON PURPOSE. If the
oracle ran different detectors, a difference in the result could not be
attributed to rule quality, which is the only thing this experiment varies.

Read the output as a ceiling, not as a target: the oracle is handed the answer
key's own rules and still cannot exceed what the detectors can express.

Usage:
    python3 oracle_gt_detect.py
"""

from __future__ import annotations

import csv
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "layers", "layer_4"))

from step5_core import (  # noqa: E402
    Rule,
    apply_strictness,
    compute_anomaly_metrics,
    compute_violation_rates,
    load_gt_windows,
    ON_DELAY_WARNING_MIN,
    merge_alarms,
    score_coverage,
    stream_and_detect,
    PLAUSIBILITY_MAX_VIOLATION_RATE,
)

GT_RULES = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed",
                        "ground_truth.csv")
OUT_DIR = _HERE


def to_float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def load_annotation_rules() -> tuple[list, dict]:
    """The annotation's own rules, read verbatim. Its sensor column already
    carries full facility identifiers, so no resolution step is required and
    none is performed -- this is the point of the experiment."""
    df = pd.read_csv(GT_RULES, dtype=str).fillna("")
    rules, skipped = [], {}
    for _, r in df.iterrows():
        cls = r["class"].strip()
        if cls == "AccessRule":                 # no telemetry detector consumes these
            skipped["AccessRule"] = skipped.get("AccessRule", 0) + 1
            continue
        if not r["sensor"].strip():
            skipped["no_sensor"] = skipped.get("no_sensor", 0) + 1
            continue
        rules.append(Rule(
            rule_id=r["ruleId"].strip(), cls=cls, sensor=r["sensor"].strip(),
            crit_hi=to_float(r["critHi"]), warn_hi=to_float(r["warnHi"]),
            warn_lo=to_float(r["warnLo"]), crit_lo=to_float(r["critLo"]),
            condition=r["condition"], action=r["action"],
            source=r["source"], severity=r["severity"].strip(),
            station=r["station"].strip()))
    return rules, skipped


def main() -> None:
    rules, skipped = load_annotation_rules()
    print(f"oracle: {len(rules)} annotated rules drive detection "
          f"(skipped: {skipped})")

    vrates = compute_violation_rates(rules)
    quarantined = {rid for rid, rate in vrates.items()
                   if rate > PLAUSIBILITY_MAX_VIOLATION_RATE}
    if quarantined:
        print(f"  plausibility guard quarantined {len(quarantined)}: "
              f"{sorted(quarantined)}")

    alarms, n_rows = stream_and_detect(rules, quarantined)
    events = merge_alarms(alarms)
    coverage = score_coverage(events, load_gt_windows())
    m = compute_anomaly_metrics(coverage, events)

    print(f"  {n_rows:,} readings -> {len(alarms):,} alarms -> {len(events)} events")
    print(f"  GT {m['anomaly_tp']}/{m['anomaly_events_total']} covered | "
          f"recall {m['anomaly_recall']:.3f} precision {m['anomaly_precision']:.3f} "
          f"F1 {m['anomaly_f1']:.3f}")
    missed = [v["gtId"] for v in coverage if v["status"] != "COVERED"]
    print(f"  missed: {missed or 'none'}")

    print("\n  per fault type:")
    df = pd.DataFrame(coverage)
    for t, g in df.groupby("type"):
        n = (g.status == "COVERED").sum()
        print(f"    {t:<14} {n}/{len(g)}")

    print("\n  under stricter acceptance:")
    strict_rows = []
    # Reported at two alarm on-delays. The shipped delay withholds
    # warning-tier alarms until a condition persists, which is sound against
    # transients but removes the opening of every episode from the covered
    # fraction; quoting one number conflates rule quality with that
    # operational filter.
    for od in (ON_DELAY_WARNING_MIN, 0):
        cov_od = coverage
        if od != ON_DELAY_WARNING_MIN:
            a, _ = stream_and_detect(rules, quarantined, on_delay_warning_min=od)
            cov_od = score_coverage(merge_alarms(a), load_gt_windows())
        print(f"    -- alarm on-delay {od} min --")
        for lbl, kw in (("any overlap (as reported)", {}),
                        (">=25% of episode covered", {"min_cov_pct": 25}),
                        (">=50% of episode covered", {"min_cov_pct": 50}),
                        ("within 30 min of onset", {"max_latency_min": 30})):
            tp, dropped = apply_strictness(cov_od, **kw)
            rec = tp / len(cov_od) if cov_od else 0.0
            print(f"    {lbl:<28} {tp:>2}/{len(cov_od)}  recall {rec:.3f}")
            strict_rows.append({"on_delay_min": od, "criterion": lbl, "tp": tp,
                                "total": len(cov_od), "recall": round(rec, 3),
                                "dropped": "|".join(dropped)})

    with open(os.path.join(OUT_DIR, "oracle_coverage.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage[0]))
        w.writeheader()
        w.writerows(coverage)
    with open(os.path.join(OUT_DIR, "oracle_strictness.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(strict_rows[0]))
        w.writeheader()
        w.writerows(strict_rows)
    print(f"\n[oracle] wrote {OUT_DIR}/oracle_coverage.csv and oracle_strictness.csv")


if __name__ == "__main__":
    main()
