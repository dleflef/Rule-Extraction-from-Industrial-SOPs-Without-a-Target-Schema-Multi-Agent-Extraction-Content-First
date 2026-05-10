"""Convert SOP documents to plain .txt for step2"""

import os
import tempfile
import zipfile
from layer_1.agent_1b_tools import convert_document_to_sop_txt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ZIP = os.path.join(_SCRIPT_DIR, "data", "seed_rules.zip")
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
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(DEFAULT_ZIP) as z:
            z.extractall(tmp)

        # Find the folder that contains the PDFs/docs
        rules_dir = None
        for root, dirs, files in os.walk(tmp):
            if any(os.path.splitext(f)[1].lower() in _SUPPORTED for f in files):
                rules_dir = root
                break

        if rules_dir is None:
            print(f"No supported documents (.pdf, .txt) found in {DEFAULT_ZIP}")
        else:
            process_dir(rules_dir)