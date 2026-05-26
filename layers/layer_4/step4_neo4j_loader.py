"""
step4_neo4j_loader.py

This script loads the iMAKS ABox (the hand-crafted ground-truth graph) into Neo4j.
It reads nodes.csv (115 nodes) and edges.csv (341 edges) from data/dataset/kg_seed/
and populates a fresh Neo4j instance. By default it wipes the existing graph first
so every run starts from a clean slate.

Run it with:
    python3 step4_neo4j_loader.py
    python3 step4_neo4j_loader.py --no-clear   (keep whatever is already in Neo4j)

The result is a Neo4j graph containing nodes of type System, Zone, Component,
Sensor, AnomalyEvent, Maintenance, Person, and SafetyEvent, connected by 341 edges.
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

NODES_CSV = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "nodes.csv")
EDGES_CSV = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "edges.csv")

NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def _clean(row: dict) -> dict:
    # Drop empty-string values so we don't push blank properties into Neo4j
    return {k: v for k, v in row.items() if v != ""}


def load_abox(driver, clear: bool = True) -> None:
    with driver.session(database=NEO4J_DATABASE) as session:

        if clear:
            # Wipe everything so we don't accumulate stale data across runs
            print("  Clearing existing graph...")
            session.run("MATCH (n) DETACH DELETE n")

        # A unique constraint on nodeId prevents duplicate nodes if the script
        # is accidentally run twice without the clear flag
        print("  Creating indexes...")
        session.run(
            "CREATE CONSTRAINT abox_node_id IF NOT EXISTS "
            "FOR (n:Node) REQUIRE n.nodeId IS UNIQUE"
        )

        # Load nodes. Every node gets the generic :Node label so we can always
        # do a label-agnostic MATCH, plus its own semantic label (e.g. :Sensor)
        print(f"  Loading nodes from {NODES_CSV} ...")
        with open(NODES_CSV, encoding="utf-8-sig") as f:
            node_rows = list(csv.DictReader(f))

        for row in node_rows:
            props = _clean(row)
            label = props.pop("label")
            session.run(
                f"MERGE (n:Node {{nodeId: $nid}}) SET n:{label} SET n += $props",
                nid=props["nodeId"],
                props=props,
            )

        print(f"    {len(node_rows)} nodes loaded")

        # Load edges. ruleRef is the only edge property we carry over from the CSV;
        # everything else is just the relationship type and the two endpoint IDs
        print(f"  Loading edges from {EDGES_CSV} ...")
        with open(EDGES_CSV, encoding="utf-8-sig") as f:
            edge_rows = list(csv.DictReader(f))

        edge_count = 0
        for row in edge_rows:
            rel_type = row["type"]
            props = {"ruleRef": row["ruleRef"]} if row.get("ruleRef") else {}
            session.run(
                f"MATCH (a:Node {{nodeId: $fid}}) "
                f"MATCH (b:Node {{nodeId: $tid}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props",
                fid=row["fromId"],
                tid=row["toId"],
                props=props,
            )
            edge_count += 1

        print(f"    {edge_count} edges loaded")


def print_summary(driver) -> None:
    # Quick sanity check after loading: print how many nodes and edges landed
    # in Neo4j, grouped by label and relationship type
    with driver.session(database=NEO4J_DATABASE) as session:
        print("\n  Node counts:")
        for rec in session.run(
            "MATCH (n) UNWIND labels(n) AS lbl "
            "WHERE lbl <> 'Node' "
            "RETURN lbl, count(*) AS cnt ORDER BY cnt DESC"
        ):
            print(f"    {rec['lbl']:20s} {rec['cnt']}")

        print("  Edge counts:")
        for rec in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC"
        ):
            print(f"    {rec['rel']:20s} {rec['cnt']}")


def main(clear: bool = True) -> None:
    # Connect, load the ABox, print a summary, then close the driver cleanly
    print(f"Neo4j URI : {NEO4J_URI}")
    print(f"Database  : {NEO4J_DATABASE}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("  Connected.\n")
        load_abox(driver, clear=clear)
        print_summary(driver)
        print("\nABox loaded successfully.")
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load iMAKS ABox into Neo4j")
    parser.add_argument("--no-clear", action="store_true", help="Keep existing graph")
    args = parser.parse_args()
    main(clear=not args.no_clear)
