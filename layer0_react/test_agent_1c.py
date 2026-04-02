"""
test_agent_1c.py

A utility script to test Agent 1C (Sensor & Signal Mapper) locally.
It extracts .zip archives or reads local files, filters for .log and .txt files,
generates the structural regex using the LLM, standardizes the logs, and saves 
the full output to an 'outputs/' directory.

Usage:
    python test_agent_1c.py
"""

import os
import sys
import json
import zipfile
import tempfile

# ── Path setup ───────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so layer_1 imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_1.agent_1c import agent_1c_app  # noqa: E402

AGENT_1C_FORMATS = {".log", ".txt"}
OUTPUT_DIR = "outputs/layer_1"


def save_results(file_path: str, final_state: dict) -> None:
    """Saves the pipeline results to a formatted JSON file in the outputs folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_agent1c.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Prepare a clean dictionary to save matching the new State keys
    output_data = {
        "source_file": os.path.basename(file_path),
        "status": final_state.get("status"),
        "error_message": final_state.get("error_message"),
        "structural_regex": final_state.get("structural_regex"),
        "standardized_logs": final_state.get("standardized_logs"),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Using default=str to safely serialize any objects
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)
        
    print(f"\nFull results saved to: {output_path}")


def run_agent_1c(file_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Testing Agent 1C on: {os.path.basename(file_path)}")
    print(f"Size: {os.path.getsize(file_path):,} bytes")
    print(f"{'='*60}")

    initial_state = {
        "file_path": file_path,
        "sample_lines": None,
        "structural_regex": None,
        "standardized_logs": None,
        "status": "pending",
        "error_message": None,
    }

    final_state = agent_1c_app.invoke(initial_state)
    status = final_state.get("status")

    print("\n--- PIPELINE RESULTS ---")
    print(f"Final status: {status}")

    if status == "error":
        print(f"[FAIL] {final_state.get('error_message')}")
        save_results(file_path, final_state)
        return

    print("\n--- 1. LLM Generated Structural Regex ---")
    print(final_state.get("structural_regex"))

    std_logs = final_state.get("standardized_logs", {})
    if std_logs:
        print(f"\n--- 2. Standardization Summary ---")
        print(f"  Total lines processed : {std_logs.get('total_lines_processed', 0):,}")
        print(f"  Successfully parsed   : {std_logs.get('parsed_records_count', 0):,}")
        print(f"  Unparsed (no match)   : {std_logs.get('unparsed_count', 0):,}")
        
        data = std_logs.get('data', [])
        if data:
            print("\n--- 3. Preview of Standardized Row 1 ---")
            print(json.dumps(data[0], indent=2))

    # Save to outputs folder
    save_results(file_path, final_state)


def test_zip_or_file(target_path: str) -> None:
    if not os.path.exists(target_path):
        print(f"[ERROR] Target path not found: {target_path}")
        return

    if target_path.lower().endswith('.zip'):
        print(f"Zip file detected. Extracting '{target_path}'...")
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(target_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            extracted_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    extracted_files.append(os.path.join(root, file))
            
            if not extracted_files:
                print("The zip file is empty.")
                return

            print(f"Found {len(extracted_files)} files in archive.")
            processed_count = 0
            
            for ext_file in extracted_files:
                filename = os.path.basename(ext_file)
                ext = os.path.splitext(filename)[1].lower()
                
                # Skip hidden files
                if filename.startswith('.'):
                    continue
                
                # Process only supported formats
                if ext in AGENT_1C_FORMATS:
                    run_agent_1c(ext_file)
                    processed_count += 1
                else:
                    print(f"\nSkipping {filename} (Unsupported format for Agent 1C).")
            
            if processed_count == 0:
                print(f"\n[INFO] No {AGENT_1C_FORMATS} files found in the archive.")
    
    else:
        ext = os.path.splitext(target_path)[1].lower()
        if ext in AGENT_1C_FORMATS:
            run_agent_1c(target_path)
        else:
            print(f"\nSkipping {os.path.basename(target_path)}. Agent 1C expects {AGENT_1C_FORMATS}.")


if __name__ == "__main__":
    # You can point this to a zip file or a direct .log file
    TARGET_DATA = os.path.join("data", "examples.zip") 
    
    # Check if we should fall back to an unzipped directory structure 
    # (useful if the user unzipped it manually like in test_agent_1a)
    if not os.path.exists(TARGET_DATA) and os.path.isdir(os.path.join("data", "examples")):
        print(f"Zip not found, scanning directory: data/examples/")
        for filename in os.listdir(os.path.join("data", "examples")):
            file_path = os.path.join("data", "examples", filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext in AGENT_1C_FORMATS:
                run_agent_1c(file_path)
    else:
        test_zip_or_file(TARGET_DATA)