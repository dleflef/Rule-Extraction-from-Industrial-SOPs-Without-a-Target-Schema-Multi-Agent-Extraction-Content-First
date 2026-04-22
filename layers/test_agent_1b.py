"""
test_agent_1b.py

A utility script to test Agent 1B (Unstructured Text Parser) locally.
It extracts .zip archives, skips structured files, processes PDFs/DOCXs,
and saves the full chunked output to an 'outputs/' directory.
"""

import os
import json
import zipfile
import tempfile
from layer_1.agent_1b import agent_1b_app

AGENT_1A_FORMATS = {".json", ".csv", ".xml", ".parquet", ".log"}
OUTPUT_DIR = "outputs/layer_1"

def save_results(file_path: str, final_state: dict):
    """Saves the pipeline results to a formatted JSON file in the outputs folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_agent1b.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Prepare a clean dictionary to save
    output_data = {
        "source_file": os.path.basename(file_path),
        "status": final_state.get("status"),
        "page_count": final_state.get("page_count"),
        "chunk_count": final_state.get("chunk_count"),
        "error_message": final_state.get("error_message"),
        "chunks": final_state.get("markdown_chunks") or []
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"Full results saved to: {output_path}")

def run_agent_1b(file_path: str):
    print(f"\n{'='*60}")
    print(f"Testing Agent 1B on: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    initial_state = {
        "file_path": file_path,
        "docling_document_dict": None,
        "page_count": None,
        "markdown_chunks": None,
        "chunk_count": None,
        "status": "pending",
        "error_message": None,
    }

    final_state = agent_1b_app.invoke(initial_state)

    print("\n--- PIPELINE RESULTS ---")
    print(f"Status:       {final_state.get('status')}")
    
    if final_state.get("status") == "error":
        print(f"Error:        {final_state.get('error_message')}")
        save_results(file_path, final_state)
        return

    print(f"Pages:        {final_state.get('page_count')}")
    print(f"Total Chunks: {final_state.get('chunk_count')}")

    # Save to outputs folder
    save_results(file_path, final_state)

def test_zip_or_file(target_path: str):
    if not os.path.exists(target_path):
        print(f"Target path not found: {target_path}")
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
            for ext_file in extracted_files:
                filename = os.path.basename(ext_file)
                ext = os.path.splitext(filename)[1].lower()
                
                if filename.startswith('.'):
                    continue
                    
                if ext in AGENT_1A_FORMATS:
                    print(f"\nSkipping {filename} due to unsupported format for Agent 1B.")
                    continue
                    
                run_agent_1b(ext_file)
    
    else:
        ext = os.path.splitext(target_path)[1].lower()
        if ext in AGENT_1A_FORMATS:
            print(f"\nSkipping {os.path.basename(target_path)} due to unsupported format for Agent 1B.")
        else:
            run_agent_1b(target_path)

if __name__ == "__main__":
    TARGET_DATA = os.path.join("layers", "data", "examples.zip")
    test_zip_or_file(TARGET_DATA)

    