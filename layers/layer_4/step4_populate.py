"""
step4_populate.py

Populates Neo4j with ONLY the LLM-extracted rules from a Layer-2 extraction
run.  No ground truth (kg_seed/nodes.csv, kg_seed/edges.csv, ground_truth.csv)
is loaded here.  The resulting graph represents *only* the knowledge the
pipeline extracted autonomously from the source documents.

Graph schema (rule subgraph)
-----------------------------
(:Station  {stationId})
(:Sensor   {sensorId, sensorType, unit})
(:Rule     {ruleId, class, sensor, station, condition, action, severity,
            critHi, warnHi, warnLo, critLo, unit, sourceFile,
            modelName, paradigm})
(:Station)-[:HAS_SENSOR]->(:Sensor)
(:Rule)-[:GOVERNS]->(:Sensor)       # only when rule.sensor is populated
(:Rule)-[:APPLIES_TO]->(:Station)   # only when rule.station is populated

Rules are stored regardless of class (including AccessRule) so the full
extraction is represented in the graph.  step4_detect.py --neo4j will
skip AccessRule and correlated rules at query time, mirroring what the
CSV-based loader does.

Run BEFORE step4b_load_abox.py (either order is fine — schemas are disjoint).

Usage
-----
    python layers/layer_4/step4_populate.py             # wipe, then load
    python layers/layer_4/step4_populate.py --no-clear  # keep existing data
    python layers/layer_4/step4_populate.py --rules <path/to/rules.csv>
"""

from __future__ import annotations

import argparse
import csv
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT    = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
STEP4_RESULTS    = os.path.join(_SCRIPT_DIR, "step4_results")

DEFAULT_RULES_CSV = os.path.join(
    _PROJECT_ROOT, "layers", "layer_2", "step2_results",
    "ext_multi_agent_langgraph_20260708_114711.csv")

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def _to_float(s: str):
    try:
        return float(s.strip())
    except (TypeError, ValueError):
        return None


def _tbl(headers: list, rows: list, title: str = "") -> None:
    if not rows:
        if title:
            print(f"\n  {title}: (none)")
        return
    widths = [
        max(len(str(h)), max(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    sep = "  +" + "+".join("-" * (w + 2) for w in widths) + "+"
    def _row(vals):
        return "  |" + "|".join(f" {str(v):<{w}} " for v, w in zip(vals, widths)) + "|"
    if title:
        print(f"\n  {title}")
    print(sep)
    print(_row(headers))
    print(sep)
    for r in rows:
        print(_row(r))
    print(sep)


def _fmt(v) -> str:
    return "" if v is None else str(v)


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_summary(rules: list[dict]) -> None:
    from collections import Counter
    os.makedirs(STEP4_RESULTS, exist_ok=True)

    # ── 1. Class distribution ────────────────────────────────────────────────
    counts = Counter(r.get("class", "?") for r in rules)
    class_rows = [[cls, cnt] for cls, cnt in sorted(counts.items())]
    _tbl(["Rule class", "Count"], class_rows, title="1. Extracted rules by class")
    _write_csv(
        os.path.join(STEP4_RESULTS, "class_distribution.csv"),
        ["rule_class", "count"],
        [{"rule_class": c, "count": n} for c, n in class_rows],
    )

    # ── 2. All rules — full dump ─────────────────────────────────────────────
    RULE_FIELDS = ["ruleId", "class", "station", "sensor", "sensorType",
                   "condition", "action", "severity",
                   "critHi", "warnHi", "warnLo", "critLo", "unit",
                   "source_file", "model_name", "paradigm"]
    _write_csv(
        os.path.join(STEP4_RESULTS, "rules_all.csv"),
        RULE_FIELDS,
        rules,
    )
    print(f"  Saved → step4_results/rules_all.csv ({len(rules)} rules)")

    # ── 3. Threshold / Operational rules — extracted bounds ──────────────────
    thr_rows = []
    thr_dicts = []
    for r in rules:
        if r.get("class", "") not in ("ThresholdRule", "OperationalRule"):
            continue
        ch = _fmt(_to_float(r.get("critHi", ""))) or "—"
        wh = _fmt(_to_float(r.get("warnHi", ""))) or "—"
        wl = _fmt(_to_float(r.get("warnLo", ""))) or "—"
        cl = _fmt(_to_float(r.get("critLo", ""))) or "—"
        thr_rows.append([
            r.get("ruleId", ""),
            r.get("sensor", "").strip() or "—",
            r.get("severity", "") or "—",
            ch, wh, wl, cl,
            r.get("unit", "") or "—",
        ])
        thr_dicts.append({
            "ruleId": r.get("ruleId", ""), "sensor": r.get("sensor", "").strip(),
            "severity": r.get("severity", ""),
            "critHi": ch, "warnHi": wh, "warnLo": wl, "critLo": cl,
            "unit": r.get("unit", ""),
        })
    _tbl(
        ["ruleId", "sensor", "severity", "critHi", "warnHi", "warnLo", "critLo", "unit"],
        thr_rows,
        title="2. ThresholdRule / OperationalRule — extracted bounds",
    )
    _write_csv(
        os.path.join(STEP4_RESULTS, "threshold_rules.csv"),
        ["ruleId", "sensor", "severity", "critHi", "warnHi", "warnLo", "critLo", "unit"],
        thr_dicts,
    )

    # ── 4. Maintenance rules ─────────────────────────────────────────────────
    maint_rows = []
    maint_dicts = []
    for r in rules:
        if r.get("class", "") != "MaintenanceRule":
            continue
        maint_rows.append([r.get("ruleId", ""), r.get("sensor", "").strip() or "—",
                           r.get("condition", "")[:70]])
        maint_dicts.append({"ruleId": r.get("ruleId", ""),
                            "sensor": r.get("sensor", "").strip(),
                            "condition": r.get("condition", ""),
                            "action": r.get("action", "")})
    _tbl(["ruleId", "sensor", "condition (truncated)"], maint_rows,
         title="3. MaintenanceRule — condition summaries")
    _write_csv(
        os.path.join(STEP4_RESULTS, "maintenance_rules.csv"),
        ["ruleId", "sensor", "condition", "action"],
        maint_dicts,
    )

    # ── 5. Sensor references ─────────────────────────────────────────────────
    sensor_refs = sorted({r.get("sensor", "").strip() for r in rules
                          if r.get("sensor", "").strip()})
    _tbl(["#", "sensor referenced by LLM"],
         [[i + 1, s] for i, s in enumerate(sensor_refs)],
         title="4. All sensor IDs referenced by extracted rules")
    _write_csv(
        os.path.join(STEP4_RESULTS, "sensor_references.csv"),
        ["sensor"],
        [{"sensor": s} for s in sensor_refs],
    )

    # ── 6. Station references ────────────────────────────────────────────────
    station_refs = sorted({r.get("station", "").strip() for r in rules
                           if r.get("station", "").strip()})
    _tbl(["#", "station referenced by LLM"],
         [[i + 1, s] for i, s in enumerate(station_refs)],
         title="5. All station IDs referenced by extracted rules")
    _write_csv(
        os.path.join(STEP4_RESULTS, "station_references.csv"),
        ["station"],
        [{"station": s} for s in station_refs],
    )

    print(f"\n  All CSVs written to layers/layer_4/step4_results/")
    print()


def load_rules(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["ruleId"] for r in rows]
    dupes = [i for i in ids if ids.count(i) > 1]
    if dupes:
        print(f"  WARNING: duplicate ruleIds: {set(dupes)}")
    print(f"  Read {len(rows)} rules from {os.path.basename(path)}")
    return rows


def populate(driver, rules: list[dict], clear: bool = True) -> None:
    sensor_rules  = [r for r in rules if r.get("sensor", "").strip()]
    station_rules = [r for r in rules if r.get("station", "").strip()]
    sensors  = sorted({r["sensor"].strip()  for r in sensor_rules})
    stations = sorted({r["station"].strip() for r in station_rules})

    with driver.session(database=NEO4J_DATABASE) as session:
        if clear:
            print("  Clearing existing graph …")
            session.run("MATCH (n) DETACH DELETE n")

        # Constraints (idempotent)
        session.run("CREATE CONSTRAINT rule_id IF NOT EXISTS "
                    "FOR (r:Rule) REQUIRE r.ruleId IS UNIQUE")
        session.run("CREATE CONSTRAINT sensor_id IF NOT EXISTS "
                    "FOR (s:Sensor) REQUIRE s.sensorId IS UNIQUE")
        session.run("CREATE CONSTRAINT station_id IF NOT EXISTS "
                    "FOR (st:Station) REQUIRE st.stationId IS UNIQUE")

        # Station nodes
        print(f"  Creating {len(stations)} Station nodes …")
        for st in stations:
            session.run("MERGE (:Station {stationId: $sid})", sid=st)

        # Sensor nodes (from rule sensor field — may have typos from LLM extraction)
        print(f"  Creating {len(sensors)} Sensor nodes …")
        for sid in sensors:
            # Find the most common sensorType/unit for this sensor across rules
            matches = [r for r in sensor_rules if r["sensor"].strip() == sid]
            stype = next((r.get("sensorType", "") for r in matches
                          if r.get("sensorType", "").strip()), "")
            unit  = next((r.get("unit", "")       for r in matches
                          if r.get("unit", "").strip()), "")
            session.run(
                "MERGE (s:Sensor {sensorId: $sid}) "
                "SET s.sensorType = $st, s.unit = $u",
                sid=sid, st=stype, u=unit)

        # Station → Sensor edges (if we can infer which station owns this sensor)
        print("  Creating HAS_SENSOR edges …")
        for r in sensor_rules:
            sid = r.get("sensor", "").strip()
            stn = r.get("station", "").strip()
            if sid and stn:
                session.run(
                    "MATCH (st:Station {stationId: $stn}), (s:Sensor {sensorId: $sid}) "
                    "MERGE (st)-[:HAS_SENSOR]->(s)",
                    stn=stn, sid=sid)

        # Rule nodes + edges
        print(f"  Creating {len(rules)} Rule nodes …")
        for r in rules:
            props = {
                "ruleId":     r.get("ruleId", ""),
                "class":      r.get("class", ""),
                "sensor":     r.get("sensor", "").strip(),
                "station":    r.get("station", "").strip(),
                "condition":  r.get("condition", ""),
                "action":     r.get("action", ""),
                "severity":   r.get("severity", ""),
                "critHi":     _to_float(r.get("critHi", "")),
                "warnHi":     _to_float(r.get("warnHi", "")),
                "warnLo":     _to_float(r.get("warnLo", "")),
                "critLo":     _to_float(r.get("critLo", "")),
                "unit":       r.get("unit", ""),
                "sourceFile": r.get("source_file", ""),
                "modelName":  r.get("model_name", ""),
                "paradigm":   r.get("paradigm", ""),
            }
            session.run(
                "MERGE (rule:Rule {ruleId: $ruleId}) SET rule += $props",
                ruleId=props["ruleId"], props=props)

            sid = props["sensor"]
            stn = props["station"]
            if sid:
                session.run(
                    "MATCH (rule:Rule {ruleId: $rid}), (s:Sensor {sensorId: $sid}) "
                    "MERGE (rule)-[:GOVERNS]->(s)",
                    rid=props["ruleId"], sid=sid)
            if stn:
                session.run(
                    "MATCH (rule:Rule {ruleId: $rid}), (st:Station {stationId: $stn}) "
                    "MERGE (rule)-[:APPLIES_TO]->(st)",
                    rid=props["ruleId"], stn=stn)

        # Summary
        n_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        n_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"  Graph: {n_nodes} nodes, {n_edges} relationships")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", default=DEFAULT_RULES_CSV,
                    help="path to the Layer-2 extraction CSV to load")
    ap.add_argument("--no-clear", dest="clear", action="store_false",
                    help="keep existing data (default: wipe graph first)")
    args = ap.parse_args()

    print(f"Connecting to Neo4j at {NEO4J_URI} …")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("  Connected.")

    rules = load_rules(args.rules)
    print_summary(rules)
    populate(driver, rules, clear=args.clear)
    driver.close()
    print("\nDone — rule subgraph loaded.  Run step4b_load_abox.py next.")


if __name__ == "__main__":
    main()
