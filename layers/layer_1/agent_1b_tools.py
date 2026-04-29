"""
layer_1/agent_1b_tools.py — Agent 1B Deterministic Tools

Purely programmatic tools that form the Agent 1B pipeline, leveraging
Docling's native structure and chunking capabilities.
Domain-agnostic: no domain-specific regexes or content-type classification
(e.g. table detection) — those are Layer 2 responsibilities.

Changes from v1
───────────────
- Removed is_table detection (Layer 2 responsibility)
- Fixed overlap splitter: loop now exits cleanly when end == len(text)
- chunk_docling_document accepts source_file for Orchestrator provenance
"""

import os
from typing import Any, Dict, List, Union

_MAX_CHUNK_CHARS: int = 2_500
_CHUNK_OVERLAP:   int = 500


# ─────────────────────────────────────────────────────────────────────────────
# Chunk splitter — string IDs to avoid collision with Docling int IDs
# ─────────────────────────────────────────────────────────────────────────────

def _split_oversized_chunk(
    chunk: Dict[str, Any], base_idx: Union[int, str]
) -> List[Dict[str, Any]]:
    """
    Splits a chunk that exceeds _MAX_CHUNK_CHARS into overlapping sub-chunks.

    Fix from v1: the loop now breaks immediately when end reaches the end of
    the text, preventing a redundant (or infinite-loop) final iteration that
    the old `start = end - _CHUNK_OVERLAP` logic could cause.
    """
    text     = chunk["content"]
    metadata = chunk["metadata"]
    results  = []
    start    = 0
    sub_idx  = 0

    while start < len(text):
        end  = min(start + _MAX_CHUNK_CHARS, len(text))
        part = text[start:end]
        results.append({
            "chunk_id":   f"{base_idx}_sub{sub_idx:02d}",
            "content":    part,
            "metadata":   metadata,
            "char_count": len(part),
        })
        sub_idx += 1

        # If we just consumed the last characters, stop.
        if end == len(text):
            break

        start = end - _CHUNK_OVERLAP

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Document Parser (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf_to_docling(file_path: str) -> Dict[str, Any]:
    """
    Converts a document into a Docling Pydantic model dictionary.

    Returns a dict with keys:
        "docling_dict" : Dict   — serialized DoclingDocument
        "page_count"   : int    — number of pages
    or:
        "error"        : str    — failure reason
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    _SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".txt"}
    if ext not in _SUPPORTED:
        return {
            "error": (
                f"Format '{ext}' not supported by Agent 1B. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        }

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import ConversionStatus

        print(f"[Agent 1B | Stage 1] Starting Docling conversion: {file_path}")
        converter = DocumentConverter()
        result = converter.convert(
            source=file_path,
            raises_on_error=False,
            max_num_pages=500,
            max_file_size=100 * 1024 * 1024,
        )

        acceptable = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}
        if result.status not in acceptable:
            errors = (
                "; ".join(e.error_message for e in result.errors)
                if result.errors
                else "Unknown"
            )
            return {
                "error": (
                    f"Conversion failed with status '{result.status}'. "
                    f"Errors: {errors}"
                )
            }

        doc_dict   = result.document.model_dump()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else None

        print(f"[Agent 1B | Stage 1] Conversion complete. Pages: {page_count}")
        return {"docling_dict": doc_dict, "page_count": page_count}

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Semantic Native Chunker (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_docling_document(
    docling_dict: Dict[str, Any],
    source_file: str,
    agent_id: str,
    processed_at: str,
) -> List[Dict[str, Any]]:
    """
    Chunks the parsed document natively, preserving headers and page numbers.

    Parameters
    ──────────
    docling_dict  : Serialized DoclingDocument from parse_pdf_to_docling.
    source_file   : Original file path — written into every chunk's metadata
                    so the Orchestrator construction log can track provenance.
    agent_id      : Identifier of this agent (e.g. "agent_1b") for the log.
    processed_at  : ISO-8601 UTC timestamp injected by chunk_node.

    Returns a list of chunk dicts, or a single error-chunk on failure.
    """
    if not docling_dict:
        return []

    try:
        from docling_core.types.doc import DoclingDocument
        from docling.chunking import HierarchicalChunker

        doc        = DoclingDocument.model_validate(docling_dict)
        chunker    = HierarchicalChunker()
        doc_chunks = list(chunker.chunk(doc))

        raw_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(doc_chunks):
            text = chunk.text.strip() if hasattr(chunk, "text") else ""
            if not text:
                continue

            headings = chunk.meta.headings if hasattr(chunk.meta, "headings") else []

            # Extract page-number provenance
            pages: List[int] = []
            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        pages.extend(
                            p.page_no for p in item.prov if hasattr(p, "page_no")
                        )

            raw_chunks.append({
                "chunk_id":   idx,
                "content":    text,
                "metadata":   {
                    "headings":     headings,
                    "page_numbers": sorted(set(pages)),
                    # ── Orchestrator construction-log fields ──────────────
                    "source_file":  source_file,
                    "agent_id":     agent_id,
                    "processed_at": processed_at,
                },
                "char_count": len(text),
            })

        # ── Context breadcrumb injection & oversized-chunk splitting ──────────
        final_chunks: List[Dict[str, Any]] = []
        n_oversized = 0

        for c in raw_chunks:
            if c["metadata"].get("headings"):
                breadcrumb   = "[Context: " + " > ".join(c["metadata"]["headings"]) + "]\n"
                c["content"]    = breadcrumb + c["content"]
                c["char_count"] = len(c["content"])

            if c["char_count"] > _MAX_CHUNK_CHARS:
                n_oversized += 1
                final_chunks.extend(_split_oversized_chunk(c, c["chunk_id"]))
            else:
                final_chunks.append(c)

        if n_oversized:
            print(
                f"[Agent 1B | Stage 2] {n_oversized} oversized chunk(s) "
                "re-split to prevent LLM overflow."
            )

        print(f"[Agent 1B | Stage 2] Produced {len(final_chunks)} chunks.")
        return final_chunks

    except Exception as exc:
        print(f"[Agent 1B | Stage 2] Native chunking failed: {exc}")
        return [{
            "chunk_id":   0,
            "content":    "Error extracting document text natively.",
            "metadata":   {"chunking_error": f"{type(exc).__name__}: {exc}"},
            "char_count": 0,
        }]