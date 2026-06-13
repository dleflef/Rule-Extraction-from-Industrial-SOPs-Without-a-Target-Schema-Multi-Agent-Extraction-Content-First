"""Convert SOP documents to plain .txt for step2"""

import os
from agent_1b_tools import convert_document_to_sop_txt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = r"C:\Users\dagha\Desktop\Agentic_KnowledgeGraph_DigitalTwins\Agentic_KnowledgeGraph_DigitalTwins\data\dataset\rules"
TEXTS_DIR = os.path.join(_SCRIPT_DIR, "texts")

_SUPPORTED = {".pdf", ".txt"}


def process_dir(folder: str) -> None:
    os.makedirs(TEXTS_DIR, exist_ok=True)
    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() in _SUPPORTED:
            try:
                txt_path = convert_document_to_sop_txt(
                    os.path.join(folder, fname), TEXTS_DIR
                )
                print(f"OK  {os.path.basename(txt_path)}  ({os.path.getsize(txt_path)} bytes)")
            except Exception as e:
                print(f"ERR {fname}: {e}")

    txt_files = sorted(f for f in os.listdir(TEXTS_DIR) if f.endswith(".txt"))
    print(f"\n{len(txt_files)} .txt file(s) in '{TEXTS_DIR}' ready for step2")


if __name__ == "__main__":
    process_dir(RULES_DIR)