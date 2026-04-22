import os
import json
import sys
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
from layers.layer_2.agent_2b import agent_2b_app

LAYER_1_DIR = "outputs/layer_1"
LAYER_2_DIR = "outputs/layer_2"
OUTPUT_DIR = "outputs/layer_2b"

def process_file(agent2a_file: str):
    """Processes a single Agent 2A output file and merges text from Layer 1."""
    try:
        # Load Entities
        with open(agent2a_file, "r", encoding="utf-8") as f:
            layer_2a_data = json.load(f)

        source_file = layer_2a_data.get("source_file")
        
        # Load Original Text Chunks from Layer 1
        base_name = os.path.splitext(source_file)[0].replace("_agent1b", "")
        layer_1_path = os.path.join(LAYER_1_DIR, f"{base_name}_agent1b.json")
        
        with open(layer_1_path, "r", encoding="utf-8") as f:
            layer_1_data = json.load(f)
            
        # Reconstruct the payload: Match text content to the entities by chunk_id
        chunks_with_entities = []
        l1_chunks = {c["chunk_id"]: c["content"] for c in layer_1_data.get("chunks", [])}
        
        for item in layer_2a_data.get("results", []):
            chunk_id = item.get("chunk_id")
            content = l1_chunks.get(chunk_id, "")
            chunks_with_entities.append({
                "chunk_id": chunk_id,
                "content": content,
                "metadata": item.get("metadata", {}),
                "entities": item.get("entities", [])
            })

        print(f"\n{'='*60}")
        print(f"Testing Agent 2B on chunks from: {source_file}")
        print(f"{'='*60}")

        initial_state = {
            "source_file": source_file,
            "chunks_with_entities": chunks_with_entities,
            "extracted_relations": None,
            "status": "pending",
            "error_message": None
        }

        # Synchronous invocation
        final_state = agent_2b_app.invoke(initial_state)

        if final_state.get("status") == "error":
            print(f"Pipeline failed for {source_file}: {final_state.get('error_message')}")
            return

        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_agent2b_triples.json")
        
        output_data = {
            "source_file": final_state["source_file"],
            "model_used": "qwen2.5-coder-7b-instruct",
            "temperature": 0.0,
            "strategy": "Zero-Shot Relation Extraction",
            "results": final_state["extracted_relations"]
        }
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Success! Triples saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {agent2a_file}: {e}")

def main():
    if not os.path.exists(LAYER_2_DIR):
        print(f"Error: Could not find Agent 2A input directory at {LAYER_2_DIR}")
        return

    json_files = glob.glob(os.path.join(LAYER_2_DIR, "*_agent2a_entities.json"))
    
    if not json_files:
        print(f"No input files found to process in {LAYER_2_DIR}")
        return

    print(f"Found {len(json_files)} files to process. Starting batch job...")
    for file_path in json_files:
        process_file(file_path)
        
    print(f"\n{'='*60}")
    print(f"Batch processing complete! Check the {OUTPUT_DIR}/ directory.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()