"""
test_agent_1a.py — Standalone test for Agent 1A: Structured Data Ingestor

Runs agent_1a_app against every .csv and .json file found in data/examples/,
printing the schema extraction and data standardization summary for each file,
and saving the full pipeline state to an 'outputs/' folder.

Usage:
    python test_agent_1a.py
"""

import json
import os
import sys
import zipfile

# ── Path setup ───────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so layer_1 imports resolve correctly
# when the script is run directly (not as part of a package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_1.agent_1a import agent_1a_app  # noqa: E402 (must come after sys.path fix)

OUTPUT_DIR = "outputs/layer_1"

def save_results(filename: str, final_state: dict) -> None:
    """Saves the pipeline results to a formatted JSON file in the outputs folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_agent1a.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Prepare a clean dictionary to save using the updated Layer 1 state keys
    output_data = {
        "source_file": filename,
        "status": final_state.get("status"),
        "error_message": final_state.get("error_message"),
        "raw_schema": final_state.get("raw_schema"),
        "standardized_data": final_state.get("standardized_data"),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Using default=str to safely serialize any lingering datetime/pandas objects
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)
        
    print(f"\nFull results saved to: {output_path}")


def run_test_on_real_data() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "data", "examples")

    if not os.path.exists(data_dir):
        zip_path = os.path.join(os.path.dirname(__file__), "data", "examples.zip")
        if not os.path.exists(zip_path):
            print(f"[ERROR] Neither '{data_dir}' nor '{zip_path}' found.")
            return
        print(f"[INFO] Unzipping '{zip_path}' into '{data_dir}'...")
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print("[INFO] Unzip complete.\n")

    # Agent 1A handles only .csv and .json — skip .pdf, .log, etc.
    test_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".csv", ".json"))
    )

    if not test_files:
        print(f"[ERROR] No .csv or .json files found in {data_dir}.")
        return

    print(f"Found {len(test_files)} structured file(s) in {data_dir}.\n")

    passed = 0
    failed = 0

    for filename in test_files:
        target_file = os.path.join(data_dir, filename)

        # Updated to match the new Agent1AState
        initial_state = {
            "file_path": target_file,
            "raw_schema": None,
            "standardized_data": None,
            "status": "pending",
            "error_message": None,
        }

        print(f"\n{'='*55}")
        print(f"  File : {filename}")
        print(f"  Size : {os.path.getsize(target_file):,} bytes")
        print("="*55)

        final_state = agent_1a_app.invoke(initial_state)
        status = final_state.get("status")

        print(f"\nFinal status: {status}")

        if status == "error":
            print(f"[FAIL] {final_state.get('error_message')}")
            save_results(filename, final_state)
            failed += 1
            continue

        # Print the new output formats
        raw_schema = final_state.get("raw_schema", {})
        print("\n--- 1. Schema Extraction ---")
        print(f"File Type: {raw_schema.get('file_type')}")
        print(f"Total Columns: {raw_schema.get('total_columns')}")
        print(f"Columns: {raw_schema.get('columns')}")

        std_data = final_state.get("standardized_data", {})
        print(f"\n--- 2. Standardization Summary ({std_data.get('total_rows', 0):,} rows processed) ---")
        print(f"Cleaned Columns: {std_data.get('cleaned_columns', [])}")
        
        # Display a single standardized record as a preview if data exists
        if std_data.get("data"):
            print("\nPreview of standardized row 1:")
            print(json.dumps(std_data["data"][0], indent=2, default=str))

        # Save to outputs folder
        save_results(filename, final_state)
        passed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_files)} file(s).")
    print("="*55)


if __name__ == "__main__":
    run_test_on_real_data()