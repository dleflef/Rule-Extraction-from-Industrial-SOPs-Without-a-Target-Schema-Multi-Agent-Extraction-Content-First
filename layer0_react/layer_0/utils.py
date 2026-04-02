import os
import json

def safe_preview_file(file_path: str, max_chars: int = 500) -> str:
    """
    Safely reads a bounded chunk of a file to determine its contents 
    without overwhelming the LLM context window.
    """
    if not os.path.exists(file_path):
        return "Error: File not found."
        
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in ['.pdf']:
            # For PDFs, would typically use PyPDF2 or pdfplumber here.
            # Returning a metadata stub for safety in Layer 0.
            return f"[Binary PDF Document] Size: {os.path.getsize(file_path)} bytes. Requires Vision/PDF parsing in Layer 1."
            
        elif ext in ['.csv', '.txt', '.json', '.log', '.html']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
                return content + ("...\n[TRUNCATED]" if len(content) == max_chars else "")
        else:
            return f"[Unsupported Format: {ext}] Size: {os.path.getsize(file_path)} bytes."
    except Exception as e:
        return f"Error reading file: {str(e)}"

def build_source_inventory(directory_path: str) -> list:
    """Scans a directory and builds the initial agnostic source inventory."""
    inventory = []
    if not os.path.exists(directory_path):
        return inventory
        
    for i, filename in enumerate(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            is_structured = ext in ['.csv', '.json']
            
            inventory.append({
                "source_id": f"SRC_{i:03d}",
                "file_path": file_path,
                "file_type": ext,
                "is_structured": is_structured,
                "status": "pending"
            })
    return inventory