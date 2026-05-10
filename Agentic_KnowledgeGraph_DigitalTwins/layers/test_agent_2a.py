"""
test_agent_2a.py

Tests Agent 2A (Entity Extractor) on Agent 1B chunk files.
Supports selecting a paradigm or running all paradigms in one go.

Usage:
    python test_agent_2a.py <agent1b_json_path> --paradigm baseline
    python test_agent_2a.py --all                    # run all paradigms on all files
    python test_agent_2a.py --paradigm cot_basic      # run one paradigm on all files
"""

import glob
import json
import os
import sys
import argparse
from layer_2.agent_2a_tools import extract_rules_from_chunk

OUTPUT_DIR = "outputs/layer_2"

# All supported paradigms (must match those in agent_2a_tools.py)
AVAILABLE_PARADIGMS = [
    "baseline", "few_shot", "graph_informed",
    "cot_basic", "cot_structured", "pre_act",
    "self_consistency", "reflexion_2turn", "reflexion_guided", "react_abox"
]


def run_extraction(agent1b_json_path: str, paradigm: str) -> dict:
    """Extract rules from a single Agent 1B file and return the output dict."""
    if not os.path.exists(agent1b_json_path):
        raise FileNotFoundError(f"File not found: {agent1b_json_path}")

    with open(agent1b_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_file = data["source_file"]
    chunks = data.get("chunks", [])
    if not chunks:
        return {
            "source_file": source_file,
            "paradigm": paradigm,
            "status": "error",
            "rule_count": 0,
            "error": "No chunks found"
        }

    all_rules = []
    for chunk in chunks:
        content = chunk.get("content", "")
        headings = chunk.get("metadata", {}).get("headings", [])
        chunk_id = chunk.get("chunk_id", "?")
        print(f"[{paradigm}] Chunk {chunk_id}: {headings}")

        rules = extract_rules_from_chunk(
            chunk_content=content,
            headings=headings,
            seed_nodes_csv="layers/data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
            paradigm=paradigm,
        )
        if rules:
            all_rules.extend(rules)

    status = "complete" if all_rules else "error"
    return {
        "source_file": source_file,
        "paradigm": paradigm,
        "status": status,
        "rule_count": len(all_rules),
        "rules": all_rules,
    }


def save_output(output_data: dict, base_name: str):
    """Save the extraction results to the layer_2 output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Append paradigm name to avoid overwriting
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_agent2a_{output_data['paradigm']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Agent 2A extraction with selectable paradigms.")
    parser.add_argument(
        "input_files", nargs="*",
        help="Specific Agent 1B JSON file(s) to process. If omitted, all *_agent1b.json in outputs/layer_1/ are used."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--paradigm", type=str, choices=AVAILABLE_PARADIGMS,
        help="Which reasoning paradigm to use."
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run all paradigms on each input file."
    )

    args = parser.parse_args()

    # Determine input files
    if args.input_files:
        input_paths = args.input_files
    else:
        input_paths = sorted(glob.glob("outputs/layer_1/*_agent1b.json"))
        if not input_paths:
            print("No agent1b JSON files found in outputs/layer_1/")
            sys.exit(1)

    paradigms_to_run = AVAILABLE_PARADIGMS if args.all else [args.paradigm]

    for path in input_paths:
        print(f"\n=== Processing {path} ===")
        for paradigm in paradigms_to_run:
            print(f"--- Paradigm: {paradigm} ---")
            try:
                output_data = run_extraction(path, paradigm)
                base_name = os.path.splitext(output_data["source_file"])[0]
                save_output(output_data, base_name)
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()


# python test_agent_2a.py --all
