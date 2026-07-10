"""
step4b_load_abox.py

Step 4b — ABox loading + rule validation.

1. Loads the physical factory ABox into Neo4j from the kg_seed CSVs:
     115 nodes  (System, Zone, Component, Sensor, AnomalyEvent,
                 Maintenance, Person, SafetyEvent)
     ~341 edges (contains, monitors, triggers, part_of, involves,
                 authorized_for, …)

2. Validates the extracted Rule nodes (loaded by step4_populate.py)
   against the real ABox Sensor nodes:
     - Creates (:Rule)-[:GOVERNS_ABOX]->(:ABoxNode:Sensor) edges where
       the rule's sensor field exactly matches the ABox sensor name.
     - Reports "active" rules (sensor found) vs "dead" rules (sensor not
       in ABox — usually LLM typos like SRV01_SERVERROM_TMP).

No ground-truth anomaly labels are used in this step.  AnomalyEvent
nodes are loaded as opaque ABox facts; their GT fields are only read
later by step5_detect.py after detection finishes.

Run AFTER step4_populate.py.

Usage
-----
    python layers/layer_4/step4b_load_abox.py           # append to existing graph
    python layers/layer_4/step4b_load_abox.py --clear   # wipe graph first (!)
"""

from __future__ import annotations

import argparse
import csv
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
RESULTS_DIR    = os.path.join(_SCRIPT_DIR, "detection_results")
STEP4_RESULTS  = os.path.join(_SCRIPT_DIR, "step4_results")

NODES_CSV = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "nodes.csv")
EDGES_CSV = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "edges.csv")

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# ABox node labels that map directly to a nodeId as primary key
ABOX_LABELS = {
    "System", "Zone", "Component", "Sensor",
    "AnomalyEvent", "Maintenance", "Person", "SafetyEvent",
}


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


def _coerce(val: str):
    """Return float if numeric-looking, else the original string, else None."""
    if not val or not val.strip():
        return None
    v = val.strip()
    try:
        return float(v)
    except ValueError:
        return v


def load_nodes(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_edges(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_props(row: dict, exclude: set[str]) -> dict:
    """Collect all non-empty fields from a CSV row as a Neo4j property dict."""
    return {k: _coerce(v) for k, v in row.items()
            if k not in exclude and v and v.strip()}


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def populate_abox(driver, nodes: list[dict], edges: list[dict],
                  clear: bool = False) -> None:
    with driver.session(database=NEO4J_DATABASE) as session:
        if clear:
            print("  Clearing entire graph …")
            session.run("MATCH (n) DETACH DELETE n")

        # Constraint on nodeId (covers all ABox node types)
        session.run("CREATE CONSTRAINT abox_node_id IF NOT EXISTS "
                    "FOR (n:ABoxNode) REQUIRE n.nodeId IS UNIQUE")

        print(f"  Creating {len(nodes)} ABox nodes …")
        by_label: dict[str, int] = {}
        for row in nodes:
            label = row.get("label", "").strip()
            if label not in ABOX_LABELS:
                print(f"    Skipping unknown label: {label}")
                continue
            node_id = row.get("nodeId", "").strip()
            if not node_id:
                continue

            props = _build_props(row, exclude={"label"})
            props["nodeId"] = node_id

            session.run(
                f"MERGE (n:ABoxNode:{label} {{nodeId: $nid}}) SET n += $props",
                nid=node_id, props=props)
            by_label[label] = by_label.get(label, 0) + 1

        for lbl, cnt in sorted(by_label.items()):
            print(f"    {cnt:3d}  :{lbl}")

        print(f"  Creating {len(edges)} ABox edges …")
        rel_counts: dict[str, int] = {}
        skipped = 0
        for row in edges:
            from_id  = row.get("fromId", "").strip()
            to_id    = row.get("toId", "").strip()
            rel_type = row.get("type", "").strip().upper().replace("-", "_")
            if not from_id or not to_id or not rel_type:
                skipped += 1
                continue
            edge_props = {k: v for k, v in row.items()
                          if k not in {"fromId", "toId", "type"} and v and v.strip()}
            session.run(
                f"MATCH (a:ABoxNode {{nodeId: $fid}}), (b:ABoxNode {{nodeId: $tid}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props",
                fid=from_id, tid=to_id, props=edge_props)
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

        for rtype, cnt in sorted(rel_counts.items()):
            print(f"    {cnt:3d}  :{rtype}")
        if skipped:
            print(f"    {skipped} edges skipped (missing fromId/toId/type)")

        n_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        n_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"  Graph total: {n_nodes} nodes, {n_edges} relationships")

    # ── Save ABox summary to step4_results/ ───────────────────────────────────
    os.makedirs(STEP4_RESULTS, exist_ok=True)
    _write_csv(
        os.path.join(STEP4_RESULTS, "abox_nodes_summary.csv"),
        ["label", "count"],
        [{"label": lbl, "count": cnt} for lbl, cnt in sorted(by_label.items())],
    )
    _write_csv(
        os.path.join(STEP4_RESULTS, "abox_edges_summary.csv"),
        ["edge_type", "count"],
        [{"edge_type": rt, "count": cnt} for rt, cnt in sorted(rel_counts.items())],
    )
    print(f"  Saved → step4_results/abox_nodes_summary.csv  "
          f"({sum(by_label.values())} nodes)")
    print(f"  Saved → step4_results/abox_edges_summary.csv  "
          f"({sum(rel_counts.values())} edges)")


def validate_rules_against_abox(driver) -> dict:
    """Link Rule nodes to real ABox Sensor nodes; surface unresolved rules.

    An "unresolved rule" is one whose sensor field does not match any ABox
    Sensor's name — almost always an LLM extraction typo.  Unresolved rules
    will never fire during detection and explain coverage gaps.

    No anomaly GT fields are read here; this is purely structural.

    Returns a dict with keys "active" (list[str] ruleIds) and
    "unresolved" (list[tuple[ruleId, sensor]]).
    """
    import re
    from collections import defaultdict

    with driver.session(database=NEO4J_DATABASE) as session:
        rule_records = list(session.run(
            "MATCH (r:Rule) "
            "WHERE r.sensor IS NOT NULL AND r.sensor <> '' "
            "RETURN r.ruleId AS ruleId, r.sensor AS sensor"
        ))

        active:      list[tuple[str, str]] = []
        unresolved:  list[tuple[str, str]] = []

        for rec in rule_records:
            rid, sensor = rec["ruleId"], rec["sensor"]
            result = session.run(
                "MATCH (r:Rule {ruleId: $rid}) "
                "MATCH (s:ABoxNode:Sensor {name: $sensor}) "
                "MERGE (r)-[:GOVERNS_ABOX]->(s) "
                "RETURN s.nodeId AS nodeId",
                rid=rid, sensor=sensor,
            ).single()
            if result:
                active.append((rid, sensor))
            else:
                unresolved.append((rid, sensor))

        nosensor_count = session.run(
            "MATCH (r:Rule) "
            "WHERE r.sensor IS NULL OR r.sensor = '' "
            "RETURN count(r) AS c"
        ).single()["c"]

        governs_count = session.run(
            "MATCH ()-[:GOVERNS_ABOX]->() RETURN count(*) AS c"
        ).single()["c"]

        total_rules = session.run(
            "MATCH (r:Rule) RETURN count(r) AS c"
        ).single()["c"]

        # How many real sensors exist in the ABox?
        abox_sensor_count = session.run(
            "MATCH (s:ABoxNode:Sensor) RETURN count(s) AS c"
        ).single()["c"]

        # Rule class for every rule (for per-class breakdown)
        class_records = list(session.run(
            "MATCH (r:Rule) RETURN r.ruleId AS ruleId, r.class AS cls"
        ))

    # ── Derived stats ─────────────────────────────────────────────────────────
    total_with_sensor  = len(active) + len(unresolved)
    active_rate        = (len(active) / total_with_sensor * 100) if total_with_sensor else 0.0
    unique_active_sensors = sorted({s for _, s in active})
    sensor_coverage_rate  = (len(unique_active_sensors) / abox_sensor_count * 100) if abox_sensor_count else 0.0
    rule_density          = (len(active) / len(unique_active_sensors)) if unique_active_sensors else 0.0

    # Per-class counts
    rule_class = {rec["ruleId"]: (rec["cls"] or "Unknown") for rec in class_records}
    active_set      = {rid for rid, _ in active}
    unresolved_set  = {rid for rid, _ in unresolved}
    class_stats: dict[str, dict] = defaultdict(lambda: {"active": 0, "unresolved": 0, "no_sensor": 0})
    for rid, _ in active:
        class_stats[rule_class.get(rid, "Unknown")]["active"] += 1
    for rid, _ in unresolved:
        class_stats[rule_class.get(rid, "Unknown")]["unresolved"] += 1
    for rec in class_records:
        rid = rec["ruleId"]
        if rid not in active_set and rid not in unresolved_set:
            class_stats[rule_class.get(rid, "Unknown")]["no_sensor"] += 1

    # Maintenance-specific resolution rate (key for Phase 2)
    maint_active     = class_stats["MaintenanceRule"]["active"]
    maint_unresolved = class_stats["MaintenanceRule"]["unresolved"]
    maint_total      = maint_active + maint_unresolved
    maint_res_rate   = (maint_active / maint_total * 100) if maint_total else 0.0

    def _diagnosis(sensor: str) -> str:
        if not re.match(r'^[A-Z0-9]+_[A-Z0-9]+_[A-Z]{2,4}$', sensor):
            return "wrong_concept"
        parts = sensor.split("_")
        station_prefix = "_".join(parts[:2]) if len(parts) >= 3 else sensor
        if any(s.startswith(station_prefix) for s in unique_active_sensors):
            return "typo_near_miss"
        return "sensor_not_in_abox"

    unresolved_with_diag = [(rid, sensor, _diagnosis(sensor)) for rid, sensor in unresolved]

    # ── Terminal tables ───────────────────────────────────────────────────────
    all_rows = sorted(
        [[rid, sensor, "ACTIVE"]     for rid, sensor in active] +
        [[rid, sensor, "UNRESOLVED"] for rid, sensor in unresolved],
        key=lambda r: r[0],
    )
    _tbl(
        ["ruleId", "sensor", "status"],
        all_rows,
        title="Rule Validation against ABox — all rules with sensor references",
    )

    # ── Summary stats table ───────────────────────────────────────────────────
    stats_rows = [
        ["Total Rule nodes loaded",                        str(total_rules)],
        ["  ├─ rules with sensor field",                   str(total_with_sensor)],
        ["  │    ├─ ACTIVE     (sensor found in ABox)",    str(len(active))],
        ["  │    └─ UNRESOLVED (sensor not in ABox)",      str(len(unresolved))],
        ["  └─ rules without sensor field",                str(nosensor_count)],
        ["─" * 40,                                         "─" * 10],
        ["Resolution rate  (sensor-carrying rules)",       f"{active_rate:.1f}%"],
        ["Sensor coverage  (ABox sensors with ≥1 rule)",   f"{len(unique_active_sensors)} / {abox_sensor_count}  ({sensor_coverage_rate:.1f}%)"],
        ["Rule density     (avg rules per covered sensor)", f"{rule_density:.2f}"],
        ["GOVERNS_ABOX edges created",                     str(governs_count)],
        ["─" * 40,                                         "─" * 10],
        ["MaintenanceRule resolution rate",                f"{maint_active}/{maint_total}  ({maint_res_rate:.1f}%)"],
    ]
    _tbl(["Metric", "Value"], stats_rows, title="Extraction Quality — Validation Summary")

    # ── Per-class breakdown table ─────────────────────────────────────────────
    cls_order = ["ThresholdRule", "OperationalRule", "MaintenanceRule", "AccessRule", "Unknown"]
    cls_rows = []
    for cls in cls_order:
        if cls not in class_stats:
            continue
        s = class_stats[cls]
        tot = s["active"] + s["unresolved"] + s["no_sensor"]
        res = (s["active"] / (s["active"] + s["unresolved"]) * 100) if (s["active"] + s["unresolved"]) else float("nan")
        res_str = f"{res:.1f}%" if res == res else "n/a"
        cls_rows.append([cls, str(tot), str(s["active"]), str(s["unresolved"]), str(s["no_sensor"]), res_str])
    _tbl(
        ["rule_class", "total", "active", "unresolved", "no_sensor", "resolution_%"],
        cls_rows,
        title="Extraction Quality — Breakdown by Rule Class",
    )

    if unresolved_with_diag:
        _tbl(
            ["ruleId", "sensor_written_by_LLM", "diagnosis"],
            [[rid, sensor, diag] for rid, sensor, diag in sorted(unresolved_with_diag)],
            title="Unresolved Rule Analysis — why each rule cannot fire",
        )
        print("  Diagnosis key:")
        print("    typo_near_miss     — station exists in ABox but sensor suffix is wrong")
        print("    sensor_not_in_abox — station prefix not recognised in ABox at all")
        print("    wrong_concept      — LLM put a concept name instead of a sensor ID")
    else:
        print("  ✓  0 unresolved rules — all sensor references resolved.")
    print(f"  ℹ  {nosensor_count} rules have no sensor field "
          f"(AccessRule / correlated — expected, skipped in detection)")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    csv_rows = [{"ruleId": r[0], "sensor": r[1], "status": r[2]} for r in all_rows]

    stats_dicts = [
        {"metric": "total_rules_loaded",            "value": total_rules},
        {"metric": "rules_with_sensor",             "value": total_with_sensor},
        {"metric": "active_rules",                  "value": len(active)},
        {"metric": "unresolved_rules",              "value": len(unresolved)},
        {"metric": "rules_without_sensor",          "value": nosensor_count},
        {"metric": "resolution_rate_pct",           "value": round(active_rate, 1)},
        {"metric": "abox_sensor_count",             "value": abox_sensor_count},
        {"metric": "unique_sensors_covered",        "value": len(unique_active_sensors)},
        {"metric": "sensor_coverage_rate_pct",      "value": round(sensor_coverage_rate, 1)},
        {"metric": "rule_density_avg",              "value": round(rule_density, 2)},
        {"metric": "governs_abox_edges",            "value": governs_count},
        {"metric": "maint_active",                  "value": maint_active},
        {"metric": "maint_unresolved",              "value": maint_unresolved},
        {"metric": "maint_resolution_rate_pct",     "value": round(maint_res_rate, 1)},
    ]

    class_dicts = [
        {
            "rule_class":      cls,
            "total":           class_stats[cls]["active"] + class_stats[cls]["unresolved"] + class_stats[cls]["no_sensor"],
            "active":          class_stats[cls]["active"],
            "unresolved":      class_stats[cls]["unresolved"],
            "no_sensor":       class_stats[cls]["no_sensor"],
            "resolution_rate_pct": round(
                class_stats[cls]["active"] / (class_stats[cls]["active"] + class_stats[cls]["unresolved"]) * 100, 1
            ) if (class_stats[cls]["active"] + class_stats[cls]["unresolved"]) else None,
        }
        for cls in cls_order if cls in class_stats
    ]

    unresolved_dicts = [
        {"ruleId": rid, "sensor_llm": sensor, "diagnosis": diag}
        for rid, sensor, diag in sorted(unresolved_with_diag)
    ]

    for out_dir in [STEP4_RESULTS, RESULTS_DIR]:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(os.path.join(out_dir, "rule_validation.csv"),
                   ["ruleId", "sensor", "status"], csv_rows)
        _write_csv(os.path.join(out_dir, "rule_validation_stats.csv"),
                   ["metric", "value"], stats_dicts)
        _write_csv(os.path.join(out_dir, "rule_class_breakdown.csv"),
                   ["rule_class", "total", "active", "unresolved", "no_sensor", "resolution_rate_pct"],
                   class_dicts)
        _write_csv(os.path.join(out_dir, "unresolved_rules_analysis.csv"),
                   ["ruleId", "sensor_llm", "diagnosis"], unresolved_dicts)

    print(f"\n  Saved → step4_results/rule_validation.csv          ({len(active)} active, {len(unresolved)} unresolved)")
    print(f"  Saved → step4_results/rule_validation_stats.csv    (14 metrics)")
    print(f"  Saved → step4_results/rule_class_breakdown.csv     ({len(cls_rows)} rule classes)")
    print(f"  Saved → step4_results/unresolved_rules_analysis.csv({len(unresolved)} unresolved rules)")
    print(f"  (all 4 also copied to detection_results/)")

    return {"active": [rid for rid, _ in active], "unresolved": unresolved}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clear", action="store_true",
                    help="wipe the entire graph before loading (default: append)")
    args = ap.parse_args()

    print(f"Connecting to Neo4j at {NEO4J_URI} …")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("  Connected.")

    nodes = load_nodes(NODES_CSV)
    edges = load_edges(EDGES_CSV)
    print(f"  Loaded {len(nodes)} nodes and {len(edges)} edges from kg_seed/")

    populate_abox(driver, nodes, edges, clear=args.clear)

    print("\nValidating extracted rules against ABox …")
    validate_rules_against_abox(driver)

    driver.close()
    print("\nDone — ABox loaded and rules validated.")


if __name__ == "__main__":
    main()
