"""
test_agent_2a.py

Tests Agent 2A (Entity Extractor) on the chunks produced by Agent 1B.
Reads a SOP_*_agent1b.json file, runs extraction on every chunk,
and saves the combined extracted rules to outputs/layer_2/.
"""

import glob
import json
import os
import sys
from layer_2.agent_2a_tools import extract_rules_from_chunk

OUTPUT_DIR = "outputs/layer_2"

def test_agent_2a(agent1b_json_path: str):
    if not os.path.exists(agent1b_json_path):
        print(f"File not found: {agent1b_json_path}")
        return

    with open(agent1b_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_file = data["source_file"]
    chunks = data.get("chunks", [])
    if not chunks:
        print("No chunks found in the input file.")
        return

    all_rules = []
    global_counters: dict = {}
    for chunk in chunks:
        content = chunk.get("content", "")
        headings = chunk.get("metadata", {}).get("headings", [])
        print(f"Processing chunk {chunk.get('chunk_id')}: {headings}")

        rules = extract_rules_from_chunk(
            chunk_content=content,
            headings=headings,
            seed_nodes_csv="layers/data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
            global_counters=global_counters,
        )
        if rules:
            all_rules.extend(rules)

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(source_file)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_agent2a.json")

    output_data = {
        "source_file": source_file,
        "status": "complete" if all_rules else "error",
        "rule_count": len(all_rules),
        "rules": all_rules,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Extraction finished. {len(all_rules)} rule(s) saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_agent_2a(sys.argv[1])
    else:
        input_files = sorted(glob.glob("outputs/layer_1/*_agent1b.json"))
        if not input_files:
            print("No agent1b JSON files found in outputs/layer_1/")
            sys.exit(1)
        for path in input_files:
            print(f"\n--- Running on {path} ---")
            test_agent_2a(path)