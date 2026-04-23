import os
import csv
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
from layers.layer_2.agent_2c import agent_2c_app

LAYER_2B_DIR   = "outputs/layer_2b"
OUTPUT_DIR     = "outputs/layer_2c"
NODES_CSV_PATH = os.path.join("layers", "extracted_seed", "kg_seed", "nodes_factory.csv")


def load_official_nodes() -> dict:
    nodes = {}
    if not os.path.exists(NODES_CSV_PATH):
        print(f"WARNING: Official nodes CSV not found at {NODES_CSV_PATH}.")
        print("Agent 2C will run but entity alignment will be skipped.")
        return nodes

    with open(NODES_CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name  = row.get("name", "").strip()
            label = row.get("label", "").strip()
            if name and label:
                nodes[name] = label

    print(f"[SeedGraphManager] Loaded {len(nodes)} official nodes for Agent 2C.")
    return nodes


def process_file(agent2b_file: str, official_nodes: dict):
    try:
        with open(agent2b_file, "r", encoding="utf-8") as f:
            layer_2b_data = json.load(f)

        source_file        = layer_2b_data.get("source_file")
        chunks_with_relations = layer_2b_data.get("results", [])

        print(f"\n{'='*60}")
        print(f"Testing Agent 2C on triples from: {source_file}")
        print(f"{'='*60}")

        initial_state = {
            "source_file":          source_file,
            "official_nodes":       official_nodes,
            "chunks_with_relations": chunks_with_relations,
            "aligned_relations":    None,
            "status":               "pending",
            "error_message":        None
        }

        final_state = agent_2c_app.invoke(initial_state)

        if final_state.get("status") == "error":
            print(f"Pipeline failed for {source_file}: {final_state.get('error_message')}")
            return

        base_name   = os.path.splitext(source_file)[0].replace("_agent1b", "")
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_agent2c_aligned.json")

        output_data = {
            "source_file": final_state["source_file"],
            "model_used":  "qwen2.5-coder-7b-instruct",
            "strategy":    "Programmatic Ontology Alignment (4-pass fuzzy matching)",
            "results":     final_state["aligned_relations"]
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        total_triples = sum(len(c.get("aligned_relations", [])) for c in final_state["aligned_relations"])
        print(f"Success! {total_triples} aligned triples saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {agent2b_file}: {e}")


def main():
    if not os.path.exists(LAYER_2B_DIR):
        print(f"Error: Agent 2B input directory not found at {LAYER_2B_DIR}")
        return

    json_files = glob.glob(os.path.join(LAYER_2B_DIR, "*_agent2b_triples.json"))

    if not json_files:
        print(f"No input files found in {LAYER_2B_DIR}")
        return

    print(f"Found {len(json_files)} files to process.")
    official_nodes = load_official_nodes()

    print("Starting batch job...")
    for file_path in json_files:
        process_file(file_path, official_nodes)

    print(f"\n{'='*60}")
    print(f"Batch processing complete. Check {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
