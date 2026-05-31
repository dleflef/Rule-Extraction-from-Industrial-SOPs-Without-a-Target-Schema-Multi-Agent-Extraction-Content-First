"""
step7_rule_to_triplets.py

This script is the final enrichment step. It takes the flat :ExtractedRule nodes
that step5 loaded and converts them into a richer typed graph structure depending
on what kind of rule each one is.

It must be run after step5 (the :ExtractedRule nodes and their :COVERS / :APPLIES_TO
edges need to already exist in Neo4j).

What gets created for each rule class:

  ThresholdRule
      :ExtractedRule -> [:HAS_SPEC] -> :ThresholdSpec  (holds the numeric limits)
      :ThresholdSpec -> [:GOVERNS]  -> :Sensor          (with threshold values on the edge)

  OperationalRule / MaintenanceRule (and any LLM sub-variants)
      :Sensor        -> [:TRIGGERS_RULE]   -> :ExtractedRule
      :ExtractedRule -> [:REQUIRES_ACTION] -> :Action  (holds action text and severity)

  AccessRule (and OccupancyRule / AcknowledgmentRule)
      :ExtractedRule -> [:RESTRICTS_ACCESS] -> :Zone

  MaintenanceRule (and PredictiveMaintenanceRule)
      :ExtractedRule -> [:SCHEDULES_MAINTENANCE] -> :Maintenance

  Cross-rule causal dependency (the GT-0009 multi-source fusion case)
      :ExtractedRule -> [:DEPENDS_ON] -> :ExtractedRule
      This is detected heuristically: if a rule's condition text mentions a station
      other than its own, it is assumed to depend on rules from that station.

Run it with:
    python3 step7_rule_to_triplets.py
    python3 step7_rule_to_triplets.py --run-file ext_ministral-3-14b_few_shot_static_run1.csv
    python3 step7_rule_to_triplets.py --clear-triplets   (remove old typed nodes/edges first)

Output: Neo4j gains :ThresholdSpec and :Action nodes plus typed edges per rule class.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

EVAL_CSV    = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "evaluation_summary.csv")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results")

NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# The ground truth uses four canonical rule classes, but LLMs sometimes invent
# more specific sub-class names (e.g. "CorrelationRule" instead of "OperationalRule").
# We group them here so that the triplet logic handles both canonical and variant names.
_THRESHOLD_CLASSES   = {"ThresholdRule"}
_OPERATIONAL_CLASSES = {"OperationalRule", "CorrelationRule", "AnomalyRule",
                         "CorrelatedFaultRule"}
_MAINTENANCE_CLASSES = {"MaintenanceRule", "PredictiveMaintenanceRule"}
_ACCESS_CLASSES      = {"AccessRule", "OccupancyRule", "AcknowledgmentRule"}

# LLM-extracted rules reference stations by their code (e.g. "ST01_FILLING"),
# but the ABox stores zones by their human-readable name (e.g. "Production Area").
# This mapping bridges the two so we can create :RESTRICTS_ACCESS edges.
_STATION_TO_ZONE: dict[str, str] = {
    "ST01_FILLING":          "Production Area",
    "ST02_SEALING":          "Production Area",
    "ST03_LABELLING":        "Production Area",
    "ST04_PACKAGING":        "Production Area",
    "SRV01_SERVERROOM":      "Server Room",
    "WRH01_WAREHOUSE":       "General Warehouse",
    "CHM01_CHEMICALSTORAGE": "Chemical Storage",
    "RND01_RDLAB":           "R&D Lab",
    "CAF01_CAFETERIA":       "Cafeteria",
}

# We detect cross-station dependencies by scanning the condition text for known
# station keywords. Each entry is (keyword_to_find, canonical_station_code).
# Both the short code (e.g. "ST01") and the long name (e.g. "FILLING") are
# included so that natural-language condition text still matches.
_STATION_KEYWORDS: list[tuple[str, str]] = [
    ("ST01", "ST01_FILLING"),    ("FILLING",   "ST01_FILLING"),
    ("ST02", "ST02_SEALING"),    ("SEALING",   "ST02_SEALING"),
    ("ST03", "ST03_LABELLING"),  ("LABELLING", "ST03_LABELLING"),
    ("ST04", "ST04_PACKAGING"),  ("PACKAGING", "ST04_PACKAGING"),
    ("SRV01", "SRV01_SERVERROOM"), ("SERVERROOM", "SRV01_SERVERROOM"),
    ("WRH01", "WRH01_WAREHOUSE"),  ("WAREHOUSE",  "WRH01_WAREHOUSE"),
    ("CHM01", "CHM01_CHEMICALSTORAGE"), ("CHEMICAL", "CHM01_CHEMICALSTORAGE"),
    ("RND01", "RND01_RDLAB"),    ("RDLAB",     "RND01_RDLAB"),
    ("CAF01", "CAF01_CAFETERIA"), ("CAFETERIA", "CAF01_CAFETERIA"),
]


def _safe_float(val: str) -> float | None:
    # LLM outputs sometimes leave threshold fields as empty strings, "nan", or
    # "None". We treat all of those as missing rather than crashing on float().
    try:
        v = float(val.strip()) if val and val.strip() not in ("", "nan", "None") else None
        return v
    except ValueError:
        return None


def _rule_uid(run_file: str, rule_id: str, idx: int) -> str:
    # Mirrors the uid formula in step5 so we can look up the right Neo4j node
    return f"{run_file}::{rule_id}::{idx}"


def _detect_foreign_stations(condition: str, own_station: str) -> list[str]:
    # Look for any station keyword in the condition text that belongs to a
    # different station than the rule itself. If found, the rule likely depends
    # on data from that foreign station (the GT-0009 pattern).
    text_upper = condition.upper()
    found: set[str] = set()
    for keyword, station in _STATION_KEYWORDS:
        if keyword in text_upper and station != own_station:
            found.add(station)
    return sorted(found)


def _norm_rule_id(rule_id: str) -> str:
    # Strip dashes and underscores and uppercase so that "RULE-MT-001" and
    # "rule_mt_001" both match the same Maintenance node in the ABox
    return re.sub(r"[-_]", "", rule_id.upper())


def add_triplets(driver, run_file: str, clear_triplets: bool = False) -> None:
    path = os.path.join(RESULTS_DIR, run_file)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return

    with open(path, encoding="utf-8") as f:
        rules = list(csv.DictReader(f))

    print(f"\n  File  : {run_file}")
    print(f"  Rules : {len(rules)}")

    # Pre-compute UIDs for every row so we don't repeat the formula in each branch.
    # Also build a station index so the DEPENDS_ON heuristic can look up rules
    # from foreign stations without scanning the full list every time.
    for i, row in enumerate(rules):
        rid = row.get("ruleId", "").strip() or f"AUTO-{i + 1}"
        row["_uid"]     = _rule_uid(run_file, rid, i)
        row["_rule_id"] = rid

    rules_by_station: dict[str, list[dict]] = {}
    for row in rules:
        st = row.get("station", "").strip()
        if st:
            rules_by_station.setdefault(st, []).append(row)

    with driver.session(database=NEO4J_DATABASE) as session:

        if clear_triplets:
            # Remove previously built typed nodes and edges so we can rebuild them
            # cleanly from a different run file without mixing data
            print("  Clearing typed triplet nodes and edges...")
            session.run("MATCH (n:ThresholdSpec) DETACH DELETE n")
            session.run("MATCH (n:Action) DETACH DELETE n")
            session.run(
                "MATCH ()-[r:GOVERNS|HAS_SPEC|TRIGGERS_RULE"
                "|REQUIRES_ACTION|RESTRICTS_ACCESS"
                "|SCHEDULES_MAINTENANCE|DEPENDS_ON]->() DELETE r"
            )

        # Indexes on uid make the MERGE lookups fast for large rule sets
        session.run(
            "CREATE INDEX threshold_spec_uid IF NOT EXISTS "
            "FOR (t:ThresholdSpec) ON (t.uid)"
        )
        session.run(
            "CREATE INDEX action_uid IF NOT EXISTS FOR (a:Action) ON (a.uid)"
        )

        c: dict[str, int] = {k: 0 for k in (
            "threshold_spec", "has_spec", "governs",
            "triggers_rule", "action_nodes", "requires_action",
            "restricts_access", "schedules_maintenance", "depends_on",
        )}

        for row in rules:
            uid        = row["_uid"]
            rule_id    = row["_rule_id"]
            rule_class = row.get("class", "").strip()
            station    = row.get("station", "").strip()
            sensor     = row.get("sensor", "").strip()
            condition  = row.get("condition", "").strip()
            action_txt = row.get("action", "").strip()
            severity   = row.get("severity", "").strip()

            # ThresholdRules get a :ThresholdSpec node that carries the numeric
            # limits (critHi, warnHi, etc.) as properties. We only create the node
            # if the LLM actually extracted at least one numeric value.
            if rule_class in _THRESHOLD_CLASSES:
                crit_hi = _safe_float(row.get("critHi", ""))
                warn_hi = _safe_float(row.get("warnHi", ""))
                warn_lo = _safe_float(row.get("warnLo", ""))
                crit_lo = _safe_float(row.get("critLo", ""))
                unit    = row.get("unit", "").strip()

                if any(v is not None for v in (crit_hi, warn_hi, warn_lo, crit_lo)):
                    spec_uid = f"SPEC::{uid}"
                    spec_props = {
                        k: v for k, v in {
                            "uid": spec_uid, "ruleId": rule_id, "unit": unit,
                            "critHi": crit_hi, "warnHi": warn_hi,
                            "warnLo": warn_lo, "critLo": crit_lo,
                        }.items() if v is not None and v != ""
                    }
                    session.run(
                        "MERGE (t:ThresholdSpec {uid: $uid}) SET t += $props",
                        uid=spec_uid, props=spec_props,
                    )
                    c["threshold_spec"] += 1

                    # Link the rule to its spec node
                    rec = session.run(
                        "MATCH (er:ExtractedRule {uid: $ruid}) "
                        "MATCH (ts:ThresholdSpec  {uid: $suid}) "
                        "MERGE (er)-[:HAS_SPEC]->(ts) RETURN count(*) AS n",
                        ruid=uid, suid=spec_uid,
                    ).single()
                    if rec:
                        c["has_spec"] += rec["n"]

                    # Also link the spec directly to the sensor it governs, and
                    # carry the threshold values on the edge for easy querying
                    if sensor:
                        edge_props = {
                            k: v for k, v in {
                                "critHi": crit_hi, "warnHi": warn_hi,
                                "warnLo": warn_lo, "critLo": crit_lo, "unit": unit,
                            }.items() if v is not None and v != ""
                        }
                        rec = session.run(
                            "MATCH (ts:ThresholdSpec {uid: $suid}) "
                            "MATCH (s:Sensor {name: $sensor}) "
                            "MERGE (ts)-[g:GOVERNS]->(s) SET g += $props "
                            "RETURN count(*) AS n",
                            suid=spec_uid, sensor=sensor, props=edge_props,
                        ).single()
                        if rec:
                            c["governs"] += rec["n"]

            # Operational and maintenance rules get a reverse edge from sensor to rule
            # (a sensor triggers a rule) plus an :Action node for the prescribed response
            if rule_class in _OPERATIONAL_CLASSES | _MAINTENANCE_CLASSES:
                if sensor:
                    rec = session.run(
                        "MATCH (s:Sensor {name: $sensor}) "
                        "MATCH (er:ExtractedRule {uid: $uid}) "
                        "MERGE (s)-[:TRIGGERS_RULE]->(er) RETURN count(*) AS n",
                        sensor=sensor, uid=uid,
                    ).single()
                    if rec:
                        c["triggers_rule"] += rec["n"]

                if action_txt:
                    action_uid = f"ACT::{uid}"
                    session.run(
                        "MERGE (a:Action {uid: $uid}) "
                        "SET a.text = $text, a.severity = $severity",
                        uid=action_uid, text=action_txt, severity=severity,
                    )
                    c["action_nodes"] += 1
                    rec = session.run(
                        "MATCH (er:ExtractedRule {uid: $ruid}) "
                        "MATCH (a:Action {uid: $auid}) "
                        "MERGE (er)-[:REQUIRES_ACTION]->(a) RETURN count(*) AS n",
                        ruid=uid, auid=action_uid,
                    ).single()
                    if rec:
                        c["requires_action"] += rec["n"]

            # Access rules point to the zone they restrict. We translate the
            # station code to a zone name via the lookup table defined above.
            if rule_class in _ACCESS_CLASSES:
                zone_name = _STATION_TO_ZONE.get(station)
                if zone_name:
                    rec = session.run(
                        "MATCH (er:ExtractedRule {uid: $uid}) "
                        "MATCH (z:Zone {name: $zone}) "
                        "MERGE (er)-[:RESTRICTS_ACCESS]->(z) RETURN count(*) AS n",
                        uid=uid, zone=zone_name,
                    ).single()
                    if rec:
                        c["restricts_access"] += rec["n"]

            # Maintenance rules are linked to their ABox Maintenance node by
            # normalised ruleId (case-insensitive, dashes/underscores removed)
            if rule_class in _MAINTENANCE_CLASSES:
                norm = _norm_rule_id(rule_id)
                rec = session.run(
                    "MATCH (er:ExtractedRule {uid: $uid}) "
                    "MATCH (m:Maintenance) "
                    "WHERE toUpper(replace(replace(m.ruleId,'-',''),'_','')) = $norm "
                    "MERGE (er)-[:SCHEDULES_MAINTENANCE]->(m) RETURN count(*) AS n",
                    uid=uid, norm=norm,
                ).single()
                if rec:
                    c["schedules_maintenance"] += rec["n"]

            # DEPENDS_ON is a heuristic: if a rule's condition mentions a station
            # other than its own, we create a dependency edge to every rule from
            # that foreign station. This is how we encode the GT-0009 causal chain
            # without requiring the LLM to explicitly declare the dependency.
            if condition:
                for foreign_station in _detect_foreign_stations(condition, station):
                    for target in rules_by_station.get(foreign_station, []):
                        if target["_uid"] == uid:
                            continue
                        rec = session.run(
                            "MATCH (src:ExtractedRule {uid: $src}) "
                            "MATCH (tgt:ExtractedRule {uid: $tgt}) "
                            "MERGE (src)-[:DEPENDS_ON]->(tgt) RETURN count(*) AS n",
                            src=uid, tgt=target["_uid"],
                        ).single()
                        if rec:
                            c["depends_on"] += rec["n"]

    print(f"\n  :ThresholdSpec nodes        : {c['threshold_spec']}")
    print(f"  :HAS_SPEC edges             : {c['has_spec']}")
    print(f"  :GOVERNS edges              : {c['governs']}")
    print(f"  :TRIGGERS_RULE edges        : {c['triggers_rule']}")
    print(f"  :Action nodes               : {c['action_nodes']}")
    print(f"  :REQUIRES_ACTION edges      : {c['requires_action']}")
    print(f"  :RESTRICTS_ACCESS edges     : {c['restricts_access']}")
    print(f"  :SCHEDULES_MAINTENANCE edges: {c['schedules_maintenance']}")
    print(f"  :DEPENDS_ON edges           : {c['depends_on']}")


def _best_run() -> str:
    # Same selection logic as step5: pick the run with the highest content_f1
    with open(EVAL_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("content_f1")]
    best = max(rows, key=lambda r: float(r["content_f1"]))
    print(f"  Best run : {best['file_name']}  F1_content={float(best['content_f1']):.4f}")
    return best["file_name"]


def main(run_file: str | None = None, clear_triplets: bool = False) -> None:
    # If no run file is specified we auto-select the best one, just like step5 does.
    # If step3 has not been run yet, we fail fast with a helpful message.
    if run_file is None:
        if os.path.exists(EVAL_CSV):
            print("Auto-selecting highest F1_content run...")
            run_file = _best_run()
        else:
            raise FileNotFoundError(
                f"Eval CSV not found: {EVAL_CSV}\n"
                "Run step3 first, or pass --run-file explicitly."
            )

    print(f"\nNeo4j URI : {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("  Connected.")
        add_triplets(driver, run_file, clear_triplets=clear_triplets)
        print("\nTriplets loaded successfully.")
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ExtractedRule nodes into typed graph triples"
    )
    parser.add_argument(
        "--run-file", default=None,
        help="Result CSV filename (default: auto-select highest F1_content run from evaluation CSV)",
    )
    parser.add_argument(
        "--clear-triplets", action="store_true",
        help="Delete ThresholdSpec/Action nodes and typed edges before loading",
    )
    args = parser.parse_args()
    main(run_file=args.run_file, clear_triplets=args.clear_triplets)
