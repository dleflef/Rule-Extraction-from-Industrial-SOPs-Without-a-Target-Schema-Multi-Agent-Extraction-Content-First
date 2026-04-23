import os
import sys
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import glob
from layers.layer_2.agent_2a import agent_2a_app



INPUT_DIR = "outputs/layer_1"
OUTPUT_DIR = "outputs/layer_2"

def process_file(file_path: str):
    """Processes a single Agent 1B output file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            layer_1_data = json.load(f)

        source_file = layer_1_data.get("source_file", os.path.basename(file_path))
        print(f"\n{'='*60}")
        print(f"Testing Agent 2A on chunks from: {source_file}")
        print(f"{'='*60}")

        prompt_mode = os.getenv("PROMPT_MODE", "graph_informed")
        initial_state = {
            "source_file":  source_file,
            "chunks":       layer_1_data.get("chunks", []),
            "prompt_mode":  prompt_mode,
            "extracted_entities": None,
            "status":       "pending",
            "error_message": None,
        }

        # Synchronous invocation
        final_state = agent_2a_app.invoke(initial_state)

        if final_state.get("status") == "error":
            print(f"Pipeline failed for {source_file}: {final_state.get('error_message')}")
            return

        base_name = os.path.splitext(source_file)[0].replace("_agent1b", "")
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_agent2a_entities.json")
        
        output_data = {
            "source_file": final_state["source_file"],
            "model_used":  "qwen2.5-coder-7b-instruct",
            "temperature": 0.0,
            "prompt_mode": prompt_mode,
            "results":     final_state["extracted_entities"],
        }
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Success! Entities saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Could not find input directory at {INPUT_DIR}")
        return

    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    
    if not json_files:
        print(f"No input files found to process in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} files to process. Starting batch job...")
    for file_path in json_files:
        process_file(file_path)
        
    print(f"\n{'='*60}")
    print("Batch processing complete! Check the outputs/layer_2/ directory.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()