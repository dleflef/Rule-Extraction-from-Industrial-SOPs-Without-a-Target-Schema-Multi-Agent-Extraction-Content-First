"""
step4_populate.py

Populates Neo4j with ONLY the LLM-extracted consensus rules
(layers/step3_results/consensus_rules.csv).  Nothing from the ground truth
(kg_seed/nodes.csv, kg_seed/edges.csv, ground_truth.csv) is loaded.
The resulting graph is the knowledge the pipeline extracted on its own.

Graph schema
------------
(:Station  {stationId})
(:Sensor   {sensorId, sensorType, unit})
(:Rule     {ruleId, class, condition, action, severity,
            critHi, warnHi, warnLo, critLo, unit, sourceFile})
(:Station)-[:HAS_SENSOR]->(:Sensor)
(:Rule)-[:GOVERNS]->(:Sensor)
(:Rule)-[:APPLIES_TO]->(:Station)

Rules with no sensor field (AccessRule, CorrelationRule, …) are stored as
Rule nodes only — no sensor/station edges are created for them.

Usage
-----
    python layers/layer_4/step4_populate.py             # wipe then load
    python layers/layer_4/step4_populate.py --no-clear  # keep existing data
"""

from __future__ import annotations

import argparse
import csv
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

CONSENSUS_CSV = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "consensus_rules.csv")

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_rules() -> list[dict]:
    with open(CONSENSUS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["ruleId"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ruleIds in consensus_rules.csv"
    return rows


def populate(driver, clear: bool = True) -> None:
    rules = load_rules()

    # Only sensor-linked rules contribute to the topology
    sensor_rules = [r for r in rules if r["sensor"]]
    sensors  = sorted({r["sensor"]  for r in sensor_rules})
    stations = sorted({r["station"] for r in sensor_rules if r["station"]})

    with driver.session(database=NEO4J_DATABASE) as session:
        if clear:
            print("  Clearing existing graph …")
            session.run("MATCH (n) DETACH DELETE n")

        session.run(
            "CREATE CONSTRAINT rule_id IF NOT EXISTS "
            "FOR (r:Rule) REQUIRE r.ruleId IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT sensor_id IF NOT EXISTS "
            "FOR (s:Sensor) REQUIRE s.sensorId IS UNIQUE"
        )

        # ── Topology nodes ─────────────────────────────────────────────────
        print(f"  Creating {len(stations)} Station nodes …")
        for st in stations:
            session.run("MERGE (:Station {stationId: $st})", st=st)

        print(f"  Creating {len(sensors)} Sensor nodes …")
        for sid in sensors:
            # pick any rule that has sensorType and unit for this sensor
            ref = next(
                (r for r in sensor_rules
                 if r["sensor"] == sid and r["sensorType"] and r["unit"]),
                next(r for r in sensor_rules if r["sensor"] == sid),
            )
            st = ref["station"]
            session.run(
                """
                MERGE (s:Sensor {sensorId: $sid})
                SET s.sensorType = $ty, s.unit = $unit
                WITH s
                MATCH (st:Station {stationId: $stId})
                MERGE (st)-[:HAS_SENSOR]->(s)
                """,
                sid=sid, ty=ref["sensorType"], unit=ref["unit"], stId=st,
            )

        # ── Rule nodes + edges ─────────────────────────────────────────────
        print(f"  Creating {len(rules)} Rule nodes …")
        for r in rules:
            props = {
                k: v
                for k, v in {
                    "ruleId":     r["ruleId"],
                    "class":      r["class"],
                    "condition":  r["condition"],
                    "action":     r["action"],
                    "severity":   r["severity"],
                    "unit":       r["unit"],
                    "sourceFile": r["source_file"],
                    "critHi":     _to_float(r["critHi"]),
                    "warnHi":     _to_float(r["warnHi"]),
                    "warnLo":     _to_float(r["warnLo"]),
                    "critLo":     _to_float(r["critLo"]),
                }.items()
                if v is not None and v != ""
            }
            session.run(
                "MERGE (r:Rule {ruleId: $rid}) SET r = $props",
                rid=r["ruleId"], props=props,
            )

            if r["sensor"]:
                session.run(
                    "MATCH (r:Rule {ruleId: $rid}), (s:Sensor {sensorId: $sid}) "
                    "MERGE (r)-[:GOVERNS]->(s)",
                    rid=r["ruleId"], sid=r["sensor"],
                )
            if r["station"] and r["station"] in stations:
                session.run(
                    "MATCH (r:Rule {ruleId: $rid}), (st:Station {stationId: $st}) "
                    "MERGE (r)-[:APPLIES_TO]->(st)",
                    rid=r["ruleId"], st=r["station"],
                )

        # ── Audit ──────────────────────────────────────────────────────────
        print("\n  Post-load node counts:")
        for rec in session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
        ):
            print(f"    {rec['label']:12} {rec['c']}")

        print("\n  Edge type counts:")
        for rec in session.run(
            "MATCH ()-[e]->() RETURN type(e) AS t, count(*) AS c ORDER BY t"
        ):
            print(f"    {rec['t']:16} {rec['c']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-clear", action="store_true",
                    help="keep existing graph contents")
    args = ap.parse_args()

    print(f"Connecting to Neo4j at {NEO4J_URI}")
    print(f"Loading consensus rules from: {os.path.relpath(CONSENSUS_CSV, _PROJECT_ROOT)}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        populate(driver, clear=not args.no_clear)
    finally:
        driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
