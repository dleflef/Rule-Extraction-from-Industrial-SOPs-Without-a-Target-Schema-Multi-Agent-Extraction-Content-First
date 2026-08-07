# layer_1/agent_1b_tools.py — robust PDF / TXT text extraction
#
# Strategy:
#   - For .pdf files: Use IBM's Docling (AI Vision) to extract layout-aware Markdown.
#   - For .txt files: read directly.
#   - Normalise text: Decode HTML, fix escaped chars, map broken ligatures.
#   - Sanitize Tables: Strip out AI bounding-box hallucinations (repeated cell text).
#   - Split into chunks using double line breaks.
#   - Save the full extracted text as a .txt file.
#
# Dependencies: docling
#   pip install docling

import html
import os
import re
from typing import List, Optional

from docling.document_converter import DocumentConverter

_SUPPORTED = {".pdf", ".txt"}
_MIN_CHUNK_CHARS = 200
_MAX_CHUNK_CHARS = 2000


def _collapse_repeated_tokens(cell: str) -> str:
    """
    Collapse a cell whose whole token sequence is one block repeated back to
    back, which is what an overlapping vision bounding box produces
    ("Inspect and lubricate HIGH Inspect and lubricate HIGH").

    The repeating unit is measured in WHOLE TOKENS, never in characters. A
    character-level rule cannot tell a duplicated bounding box from a letter
    that legitimately ends one word and begins the next, so it silently eats
    characters out of ordinary text: "Conveyor speed drop" -> "Conveyor
    speedrop", "SRV01_SERVERRO OM" -> "SRV01_SERVERROM". Matching whole
    tokens has no such failure mode, in any language or document layout.
    """
    tokens = cell.split()
    n = len(tokens)
    if n < 2:
        return cell

    # Smallest period p (a proper divisor of n) whose block tiles the sequence.
    for p in range(1, n // 2 + 1):
        if n % p:
            continue
        if all(tokens[i] == tokens[i % p] for i in range(n)):
            return " ".join(tokens[:p])

    return cell


def _deduplicate_table_cells(text: str) -> str:
    """
    Scans for Markdown table rows and removes repeated substring hallucinations 
    caused by overlapping vision-model bounding boxes.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Identify markdown table rows
        if line.strip().startswith('|') and line.strip().endswith('|'):
            cells = line.split('|')
            cleaned_cells = []
            
            for cell in cells:
                c = _collapse_repeated_tokens(cell.strip())

                # Re-pad the cell with spaces for clean markdown formatting
                cleaned_cells.append(f" {c} " if c else "")
                
            cleaned_lines.append('|'.join(cleaned_cells))
        else:
            cleaned_lines.append(line)
            
    return '\n'.join(cleaned_lines)


def _normalise(text: str) -> str:
    """Decode HTML entities, remove escaped underscores, and fix known artifacts."""
    text = html.unescape(text)
    text = text.replace("\\_", "_")
    
    # FIX: Font encoding issues causing ligatures to replace arrows.
    # 'fi' was mapped to right-arrow (→)
    # 'fl' was mapped to down-arrow (↓) or trend-down indicators based on SOP_003
    text = re.sub(r'\bfi\b', '→', text)
    text = re.sub(r'\bfl\b', '↓', text)
    
    # Clean up repeated table cell content caused by Docling AI overlap
    text = _deduplicate_table_cells(text)

    # Clean up excessive spacing within lines to keep tokens low
    text = re.sub(r' {2,}', ' ', text)
    
    return text


def _split_into_chunks(text: str) -> List[str]:
    """Split text at double newlines, merging until size limits are reached.
    This works exceptionally well with Markdown format."""
    paragraphs = re.split(r"\n{2,}", text)
    buffer = ""
    result: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) > _MAX_CHUNK_CHARS and buffer:
            result.append(buffer.strip())
            buffer = para
        else:
            buffer = candidate

    if buffer.strip():
        result.append(buffer.strip())

    # Merge tiny last chunk into previous
    if len(result) >= 2 and len(result[-1]) < _MIN_CHUNK_CHARS:
        last = result.pop()
        result[-1] = result[-1] + "\n\n" + last

    return result


def _extract_pdf_markdown(file_path: str) -> str:
    """Extract layout-aware text and tables as Markdown using Docling."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


def convert_document_to_sop_txt(
    file_path: str,
    texts_dir: str = "texts",
    chunks: Optional[List[str]] = None,
) -> str:
    """
    Convert a document to a plain-text/markdown .txt file saved in texts_dir.
    """
    if chunks is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _SUPPORTED:
            raise ValueError(
                f"Unsupported format '{ext}'. Only .pdf and .txt are supported."
            )

        if ext == ".pdf":
            raw_text = _extract_pdf_markdown(file_path)
            full_text = _normalise(raw_text)
            chunks = _split_into_chunks(full_text)
        else:  # .txt
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            full_text = _normalise(raw_text)
            chunks = _split_into_chunks(full_text)

    # Join the chunks back together to save the full document
    full_text = "\n\n".join(chunks)

    os.makedirs(texts_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(texts_dir, f"{stem}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return out_path