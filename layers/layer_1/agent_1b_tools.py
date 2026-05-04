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

# Supported file extensions that Docling can handle
_SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".txt"}
# Minimum number of characters a chunk must have before merging
_MIN_CHUNK_CHARS = 200
# Maximum number of characters a chunk may contain before splitting
_MAX_CHUNK_CHARS = 4000


def _normalise(text: str) -> str:
    """Decode HTML entities and remove backslash‑escaped underscores."""
    # Replace HTML entities like &amp; with their actual characters
    text = html.unescape(text)
    # Remove the backslash before underscores (e.g., \_ -> _)
    text = text.replace("\\_", "_")
    return text


def _split_into_chunks(text: str) -> List[str]:
    """Split text at double newlines, merging until size limits are reached."""
    # Split the text wherever there are two or more consecutive newlines
    paragraphs = re.split(r"\n{2,}", text)
    buffer = ""                 # accumulates paragraphs until chunk is big enough
    result: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue            # skip empty paragraphs

        # If we already have a buffer, try to see if adding this paragraph would
        # exceed the maximum chunk size
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) > _MAX_CHUNK_CHARS and buffer:
            # Current buffer is large enough -> store it and start a new one
            result.append(buffer.strip())
            buffer = para
        else:
            # Otherwise keep building the current chunk
            buffer = candidate

    # Don't forget the last accumulated chunk
    if buffer.strip():
        result.append(buffer.strip())

    # If the last chunk is tiny (smaller than MIN_CHUNK_CHARS), merge it into the
    # previous one to avoid giving the LLM an almost empty piece of text
    if len(result) >= 2 and len(result[-1]) < _MIN_CHUNK_CHARS:
        last = result.pop()
        result[-1] = result[-1] + "\n\n" + last

    return result


def parse_pdf_to_markdown_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Convert a document to a list of plain‑text chunks.

    Returns list of dicts with: chunk_id, content, metadata (minimal), char_count.
    """
    # Ensure the Docling library is available
    if not _HAS_DOCLING:
        raise ImportError("Docling is required. Install with: pip install docling")
    # Check that the file exists on disk
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # Extract file extension and verify it's a supported format
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _SUPPORTED:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {sorted(_SUPPORTED)}")

    # Use Docling to convert the document to a structured representation
    converter = DocumentConverter()
    result = converter.convert(file_path)

    # Take ONLY the raw Markdown – no table repair, no DataFrame export.
    markdown_text = result.document.export_to_markdown()
    # Clean up common artefacts from the conversion
    markdown_text = _normalise(markdown_text)

    # Split the cleaned Markdown into size‑controlled chunks
    raw_chunks = _split_into_chunks(markdown_text)

    # Build the final list of chunk dictionaries
    chunks = []
    for idx, content in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": idx,
            "content": content,
            "metadata": {
                "headings": [],        # intentionally left empty – the LLM will interpret headings itself
                "page_numbers": [],
            },
            "char_count": len(content),
        })

    return chunks