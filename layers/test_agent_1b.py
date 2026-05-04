"""
test_agent_1b.py

Minimal test for Agent 1B: converts PDFs/DOCXs to Markdown chunks
and saves the output in outputs/layer_1/.
"""

import os
import json
import zipfile
import tempfile
from layer_1.agent_1b_tools import parse_pdf_to_markdown_chunks

OUTPUT_DIR = "outputs/layer_1"

def run_agent_1b(file_path: str):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        chunks = parse_pdf_to_markdown_chunks(file_path)
        status = "complete"
        error = None
    except Exception as e:
        chunks = []
        status = "error"
        error = str(e)

    print(f"Status:  {status}")
    if error:
        print(f"Error:   {error}")

    print(f"Chunks:  {len(chunks)}")
    for i, chunk in enumerate(chunks):
        head = chunk["metadata"].get("headings", [])
        print(f"  [{i}] {' > '.join(head)} — {chunk['char_count']} chars")

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_agent1b.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": os.path.basename(file_path),
            "status": status,
            "error_message": error,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }, f, indent=2, ensure_ascii=False)

    print(f"Output saved to: {out_path}")

def process_path(target: str):
    if not os.path.exists(target):
        print(f"Not found: {target}")
        return

    if target.lower().endswith(".zip"):
        print(f"Extracting zip: {target}")
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(target, "r") as zf:
                zf.extractall(tmp)
            for root, _, files in os.walk(tmp):
                for f in files:
                    if not f.startswith("."):
                        run_agent_1b(os.path.join(root, f))
    else:
        run_agent_1b(target)

if __name__ == "__main__":
    # Adjust path as needed – here it expects a folder or zip in layers/data
    TARGET = os.path.join("layers", "data", "examples.zip")
    process_path(TARGET)