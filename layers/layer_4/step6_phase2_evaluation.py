"""
step6_phase2_evaluation.py

This is the phase 2 evaluation: given the :ExtractedRule nodes already loaded
into Neo4j by step5, we check how well they cover the real events in the ABox.

For each AnomalyEvent (14 total) and Maintenance node (8 total) we ask: does
at least one extracted rule govern the sensor that triggers this event? If yes
the event is COVERED; otherwise it is a GAP.

GT-0009 gets its own binary test because it is a multi-source fusion case.
A speed anomaly at ST04_PACKAGING is caused by a current drift at ST02_SEALING
with a 90-minute lag. Passing requires an extracted rule that covers
ST04_PACKAGING_SPD AND whose condition text mentions the ST02/sealing origin.

Run it with:
    python3 step6_phase2_evaluation.py

Output: a console report and layers/step3_results/phase2_evaluation.csv
"""

from __future__ import annotations

import csv
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
OUTPUT_FILE   = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "phase2_evaluation.csv")

NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# 70% is the minimum acceptable coverage stated in the dataset guide.
# If the extracted rules cover fewer ABox events than this, the pipeline has failed.
MIN_COVERAGE = 0.70


# ── QUERIES ───────────────────────────────────────────────────────────────────

def eval_anomaly_events(session) -> list[dict]:
    # Walk from each AnomalyEvent back to its sensor, then check whether any
    # ExtractedRule covers that sensor. We collect the first few matching ruleIds
    # so we can see which rules are doing the covering in the output.
    records = session.run("""
        MATCH (s:Sensor)-[:triggers]->(ae:AnomalyEvent)
        OPTIONAL MATCH (r:ExtractedRule)-[:COVERS]->(s)
        RETURN
            ae.gtId          AS gtId,
            ae.anomalyType   AS anomalyType,
            ae.severity      AS severity,
            s.name           AS sensor,
            count(r)         AS coveringRules,
            collect(r.ruleId)[0..3] AS matchedIds
        ORDER BY ae.gtId
    """)

    results = []
    for rec in records:
        status = "COVERED" if rec["coveringRules"] > 0 else "GAP"
        results.append({
            "event_id":       rec["gtId"],
            "type":           "AnomalyEvent",
            "anomaly_type":   rec["anomalyType"],
            "severity":       rec["severity"],
            "sensor":         rec["sensor"],
            "covering_rules": rec["coveringRules"],
            "matched_ids":    ", ".join(rec["matchedIds"] or []),
            "status":         status,
        })
    return results


def eval_maintenance(session) -> list[dict]:
    # Maintenance nodes are linked to AnomalyEvents (via :triggers) which in turn
    # are linked to Sensors. We match coverage by asking: does any ExtractedRule
    # of class MaintenanceRule cover the sensor that feeds into this maintenance node?
    # This is more reliable than ruleId string matching because LLM-extracted IDs
    # use different naming conventions than the ABox (e.g. RULE-ST02-03 vs MAINT-02).
    records = session.run("""
        MATCH (s:Sensor)-[:triggers]->(m:Maintenance)
        OPTIONAL MATCH (r:ExtractedRule)-[:COVERS]->(s)
        WITH m, s, [x IN collect(r)
                    WHERE x.ruleClass IN ['MaintenanceRule', 'PredictiveMaintenanceRule']]
                   AS maintRules
        RETURN
            m.ruleId              AS maintId,
            m.priority            AS priority,
            s.name                AS sensor,
            size(maintRules)      AS coveringRules,
            [x IN maintRules | x.ruleId][0..3] AS matchedIds
        ORDER BY m.ruleId
    """)

    results = []
    for rec in records:
        status = "COVERED" if rec["coveringRules"] > 0 else "GAP"
        results.append({
            "event_id":       rec["maintId"],
            "type":           "Maintenance",
            "anomaly_type":   "",
            "severity":       rec["priority"] or "",
            "sensor":         rec["sensor"] or "",
            "covering_rules": rec["coveringRules"],
            "matched_ids":    ", ".join(rec["matchedIds"] or []),
            "status":         status,
        })
    return results


def eval_gt0009(session) -> dict:
    # GT-0009 is the hardest test case in the dataset: it requires the LLM to
    # understand a causal chain that spans two stations. Passing means the model
    # extracted a rule that (1) covers the packaging speed sensor AND (2) mentions
    # the sealing station as the upstream cause. Both conditions must hold.

    rec = session.run("""
        MATCH (r:ExtractedRule)-[:COVERS]->(s:Sensor {name: 'ST04_PACKAGING_SPD'})
        RETURN count(r) AS cnt, collect(r.ruleId)[0..5] AS ids
    """).single()

    covers_spd = (rec["cnt"] > 0) if rec else False
    causal_rule = None

    if covers_spd:
        rec2 = session.run("""
            MATCH (r:ExtractedRule)-[:COVERS]->(s:Sensor {name: 'ST04_PACKAGING_SPD'})
            WHERE toLower(r.condition) CONTAINS 'st02'
               OR toLower(r.condition) CONTAINS 'sealing'
               OR toLower(r.ruleId)    CONTAINS 'st02'
            RETURN r.ruleId AS ruleId, r.condition AS condition
            LIMIT 1
        """).single()
        if rec2:
            causal_rule = rec2["ruleId"]

    return {
        "passed":       covers_spd and causal_rule is not None,
        "covers_spd":   covers_spd,
        "causal_rule":  causal_rule,
    }


# ── REPORTING ─────────────────────────────────────────────────────────────────

def _flag(status: str) -> str:
    # Simple visual indicator for the console report
    return "✓" if status == "COVERED" else "✗"


def print_report(ae: list[dict], maint: list[dict], gt0009: dict) -> None:
    # Print a human-readable summary of the coverage results to the console
    sep = "=" * 65
    print(f"\n{sep}")
    print("PHASE 2 EVALUATION — iMAKS")
    print(sep)

    ae_covered = sum(1 for r in ae if r["status"] == "COVERED")
    print(f"\n  AnomalyEvents  {ae_covered}/{len(ae)} COVERED")
    for r in ae:
        print(
            f"    {_flag(r['status'])} {r['event_id']:8s} "
            f"{r['anomaly_type']:15s} {r['sensor']:30s} → {r['status']}"
        )

    maint_covered = sum(1 for r in maint if r["status"] == "COVERED")
    print(f"\n  Maintenance    {maint_covered}/{len(maint)} COVERED")
    for r in maint:
        print(
            f"    {_flag(r['status'])} {r['event_id']:10s} "
            f"{r['sensor']:30s} → {r['status']}"
        )

    total = len(ae) + len(maint)
    covered = ae_covered + maint_covered
    pct = 100 * covered / total if total else 0
    target_ok = "✓" if pct >= MIN_COVERAGE * 100 else "✗"
    print(f"\n  Overall coverage : {covered}/{total} ({pct:.1f}%)  target ≥{MIN_COVERAGE*100:.0f}% {target_ok}")

    verdict = "PASS ✓" if gt0009["passed"] else "FAIL ✗"
    print(f"\n  GT-0009 (multi-source fusion) : {verdict}")
    print(f"    Covers ST04_PACKAGING_SPD    : {gt0009['covers_spd']}")
    print(f"    Causal rule (ST02 ref) found : {gt0009['causal_rule']}")
    print(sep)


def save_csv(ae: list[dict], maint: list[dict]) -> None:
    # Persist the per-event results so they can be used in tables and figures
    # in the paper without having to re-run Neo4j queries
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fields = ["event_id", "type", "anomaly_type", "severity", "sensor",
              "covering_rules", "matched_ids", "status"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ae + maint)
    print(f"\n  Saved: {OUTPUT_FILE}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Run all three checks (anomaly events, maintenance, GT-0009), print the
    # report to the console, and save the per-event results to CSV
    print(f"Neo4j URI : {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("  Connected.\n")
        with driver.session(database=NEO4J_DATABASE) as session:
            ae     = eval_anomaly_events(session)
            maint  = eval_maintenance(session)
            gt0009 = eval_gt0009(session)
        print_report(ae, maint, gt0009)
        save_csv(ae, maint)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
