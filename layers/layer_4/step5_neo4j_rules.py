"""
step5_neo4j_rules.py

This script takes the best LLM extraction run and loads it into Neo4j as
:ExtractedRule nodes. "Best" means the run with the highest f1_content score
from the step3 evaluation CSV.

For each extracted rule we create a node and then try to wire it to the
matching :Sensor (via :COVERS) and to the matching :Component (via :APPLIES_TO).
These connections are what allow step6 and step7 to evaluate coverage and build
the typed triplet structure downstream.

Run it with:
    python3 step5_neo4j_rules.py
    python3 step5_neo4j_rules.py --run-file ext_ministral-3-14b_few_shot_static_run1.csv
    python3 step5_neo4j_rules.py --clear-rules   (wipe existing ExtractedRule nodes first)
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

EVAL_CSV        = os.path.join(_PROJECT_ROOT, "layers", "step3_results", "comprehensive_evaluation_results.csv")
RESULTS_DIR     = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results")
DEFAULT_RUN_FILE = "ext_ministral-3-14b_few_shot_static_run1.csv"

NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# These are bookkeeping columns added by the step2 pipeline runner.
# We strip them before pushing properties to Neo4j so the node stays clean.
_META_FIELDS = {"source_file", "model_name", "paradigm", "level", "run_id", "llm_turns", "text_truncated"}


def best_run() -> str:
    # Read the step3 evaluation results and pick whichever run scored highest
    # on f1_content. That is the run we will load into Neo4j.
    with open(EVAL_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("f1_content")]
    best = max(rows, key=lambda r: float(r["f1_content"]))
    print(
        f"  Best run : {best['file']}\n"
        f"  F1_content={float(best['f1_content']):.4f}  "
        f"P={float(best['f1_content_precision']):.4f}  "
        f"R={float(best['f1_content_recall']):.4f}"
    )
    return best["file"]


def _parse_meta(filename: str) -> dict:
    # The result filenames follow the pattern:
    # ext_<model>_<paradigm>_run<N>.csv
    # We parse them here so we can tag each :ExtractedRule node with the
    # model and paradigm that produced it, which is useful for later analysis.
    # Paradigms can be 1, 2, or 3 tokens (e.g. "naive" vs "few_shot_static"),
    # so we try each length until we get a non-empty model string.
    stem = filename.replace("ext_", "").replace(".csv", "")
    parts = stem.split("_")
    run_part = next((p for p in reversed(parts) if p.startswith("run")), "run1")
    run_idx = parts.index(run_part)
    for paradigm_len in (3, 2, 1):
        if run_idx >= paradigm_len:
            paradigm = "_".join(parts[run_idx - paradigm_len: run_idx])
            model = "_".join(parts[: run_idx - paradigm_len])
            if model:
                return {"model": model, "paradigm": paradigm, "run_n": run_part}
    return {"model": stem, "paradigm": "unknown", "run_n": "run1"}


def load_rules(driver, run_file: str, clear_rules: bool = False) -> None:
    path = os.path.join(RESULTS_DIR, run_file)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return

    with open(path, encoding="utf-8") as f:
        rules = list(csv.DictReader(f))

    meta = _parse_meta(run_file)
    print(f"\n  File      : {run_file}")
    print(f"  Model     : {meta['model']}")
    print(f"  Paradigm  : {meta['paradigm']}")
    print(f"  Rules     : {len(rules)}")

    with driver.session(database=NEO4J_DATABASE) as session:
        if clear_rules:
            # Remove old extracted rules so we don't mix results from different runs
            print("  Clearing existing ExtractedRule nodes...")
            session.run("MATCH (n:ExtractedRule) DETACH DELETE n")

        # An index on uid makes the MERGE operations fast
        session.run(
            "CREATE INDEX extracted_rule_id IF NOT EXISTS FOR (r:ExtractedRule) ON (r.uid)"
        )

        covers_count  = 0
        applies_count = 0

        for i, row in enumerate(rules):
            rule_id = row.get("ruleId", "").strip() or f"AUTO-{i+1}"
            # The uid combines the filename, ruleId, and row index so it is
            # globally unique even if two runs produce rules with the same ruleId
            uid = f"{run_file}::{rule_id}::{i}"

            # Keep only content fields on the node; drop the step2 bookkeeping columns
            props = {k: v for k, v in row.items() if v.strip() and k not in _META_FIELDS}
            props.update(
                uid=uid,
                ruleId=rule_id,
                ruleClass=row.get("class", ""),
                model=meta["model"],
                paradigm=meta["paradigm"],
                run_file=run_file,
            )

            session.run(
                "MERGE (r:ExtractedRule {uid: $uid}) SET r += $props",
                uid=uid,
                props=props,
            )

            # Wire the rule to the sensor it governs (if the LLM named one and
            # that sensor actually exists in the ABox)
            sensor = row.get("sensor", "").strip()
            if sensor:
                rec = session.run(
                    "MATCH (r:ExtractedRule {uid: $uid}) "
                    "MATCH (s:Sensor {name: $sensor}) "
                    "MERGE (r)-[:COVERS]->(s) RETURN count(*) AS n",
                    uid=uid,
                    sensor=sensor,
                ).single()
                if rec:
                    covers_count += rec["n"]

            # Wire the rule to its station component in the ABox
            station = row.get("station", "").strip()
            if station:
                rec = session.run(
                    "MATCH (r:ExtractedRule {uid: $uid}) "
                    "MATCH (c:Component {name: $station}) "
                    "MERGE (r)-[:APPLIES_TO]->(c) RETURN count(*) AS n",
                    uid=uid,
                    station=station,
                ).single()
                if rec:
                    applies_count += rec["n"]

        print(f"\n  :ExtractedRule nodes   : {len(rules)}")
        print(f"  :COVERS edges          : {covers_count}")
        print(f"  :APPLIES_TO edges      : {applies_count}")
        unmatched_sensors  = len(rules) - covers_count
        unmatched_stations = len(rules) - applies_count
        print(f"  Unmatched sensors      : {unmatched_sensors}")
        print(f"  Unmatched stations     : {unmatched_stations}")


def main(run_file: str | None = None, clear_rules: bool = False) -> None:
    # If no specific run file is given, auto-pick the best one from step3 results.
    # Fall back to the hardcoded default only if the evaluation CSV does not exist yet.
    if run_file is None:
        if os.path.exists(EVAL_CSV):
            print("Auto-selecting highest F1_content run...")
            run_file = best_run()
        else:
            run_file = DEFAULT_RUN_FILE
            print(f"Eval CSV not found, using default: {run_file}")

    print(f"\nNeo4j URI : {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("  Connected.")
        load_rules(driver, run_file, clear_rules=clear_rules)
        print("\nExtractedRule nodes loaded successfully.")
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load best extracted rules into Neo4j")
    parser.add_argument("--run-file", default=None, help=f"Result CSV filename (default: {DEFAULT_RUN_FILE})")
    parser.add_argument("--clear-rules", action="store_true", help="Delete existing ExtractedRule nodes first")
    args = parser.parse_args()
    main(run_file=args.run_file, clear_rules=args.clear_rules)
