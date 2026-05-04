# layer_1/agent_1b_tools.py — format‑agnostic document chunker
#
# Strategy:
#   1. Convert any document to plain Markdown with Docling.
#   2. Normalise text (HTML entities, escaped underscores).
#   3. Split into chunks using ONLY paragraph breaks (no domain patterns).
#
# The LLM receives the text as it is and does all interpretation.

import html
import os
import re
from typing import Any, Dict, List

try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True
except ImportError:
    _HAS_DOCLING = False

_SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".txt"}
_MIN_CHUNK_CHARS = 200
_MAX_CHUNK_CHARS = 4000


def _normalise(text: str) -> str:
    """Decode HTML entities and remove backslash‑escaped underscores."""
    text = html.unescape(text)
    text = text.replace("\\_", "_")
    return text


def _split_into_chunks(text: str) -> List[str]:
    """Split text at double newlines, merging until size limits are reached."""
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

    # Merge a tiny trailing chunk into the previous one.
    if len(result) >= 2 and len(result[-1]) < _MIN_CHUNK_CHARS:
        last = result.pop()
        result[-1] = result[-1] + "\n\n" + last

    return result


def parse_pdf_to_markdown_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Convert a document to a list of plain‑text chunks.

    Returns list of dicts with: chunk_id, content, metadata (minimal), char_count.
    """
    if not _HAS_DOCLING:
        raise ImportError("Docling is required. Install with: pip install docling")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _SUPPORTED:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {sorted(_SUPPORTED)}")

    converter = DocumentConverter()
    result = converter.convert(file_path)

    # Take ONLY the raw Markdown – no table repair, no DataFrame export.
    markdown_text = result.document.export_to_markdown()
    markdown_text = _normalise(markdown_text)

    raw_chunks = _split_into_chunks(markdown_text)

    chunks = []
    for idx, content in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": idx,
            "content": content,
            "metadata": {
                "headings": [],
                "page_numbers": [],
            },
            "char_count": len(content),
        })

    return chunks