"""
layer_1/agent_1b_tools.py — Agent 1B Deterministic Tools

Implements the purely programmatic tools that form the Agent 1B pipeline, 
leveraging Docling's native structure and chunking capabilities. 
Stripped of domain-specific regexes to remain completely agnostic.
"""

import os
from typing import Any, Dict, List, Union

_MAX_CHUNK_CHARS: int = 2_500
_CHUNK_OVERLAP:   int = 100

# ─────────────────────────────────────────────────────────────────────────────
# Chunk splitter — string IDs to prevent collision with Docling ints
# ─────────────────────────────────────────────────────────────────────────────

def _split_oversized_chunk(
    chunk: Dict[str, Any], base_idx: Union[int, str]
) -> List[Dict[str, Any]]:
    """Splits chunks that exceed the token/character limit to prevent LLM overflow."""
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
        start    = end - _CHUNK_OVERLAP
        if start >= len(text):
            break

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Document Parser (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf_to_docling(file_path: str) -> Dict[str, Any]:
    """Converts a document into a Docling Pydantic model dictionary."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    _SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".txt"}
    if ext not in _SUPPORTED:
        return {"error": f"Format '{ext}' not supported by Agent 1B. Supported: {sorted(_SUPPORTED)}."}

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

        acceptable_statuses = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}
        if result.status not in acceptable_statuses:
            errors = (
                "; ".join([e.error_message for e in result.errors])
                if result.errors
                else "Unknown"
            )
            return {"error": f"Conversion failed with status '{result.status}'. Errors: {errors}"}

        doc_dict   = result.document.model_dump()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else None

        print(f"[Agent 1B | Stage 1] Conversion complete. Pages: {page_count}")
        return {"docling_dict": doc_dict, "page_count": page_count}

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Semantic Native Chunker (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_docling_document(docling_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunks the parsed document natively, preserving headers and page numbers."""
    if not docling_dict:
        return []

    try:
        from docling_core.types.doc import DoclingDocument
        from docling.chunking import HierarchicalChunker

        doc     = DoclingDocument.model_validate(docling_dict)
        chunker = HierarchicalChunker()
        doc_chunks = list(chunker.chunk(doc))

        raw_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(doc_chunks):
            text = chunk.text.strip() if hasattr(chunk, "text") else ""
            if not text:
                continue

            headings = chunk.meta.headings if hasattr(chunk.meta, "headings") else []
            pages: List[int] = []
            
            # Extract provenance (page numbers)
            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        pages.extend([p.page_no for p in item.prov if hasattr(p, "page_no")])

            # Detect table chunks: try Docling class first, fall back to markdown markers
            is_table = False
            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if item.__class__.__name__ == "TableItem":
                        is_table = True
                        break
            if not is_table and text:
                pipe_lines = sum(1 for ln in text.split("\n") if ln.strip().startswith("|"))
                is_table = pipe_lines >= 2

            raw_chunks.append({
                "chunk_id":   idx,
                "content":    text,
                "metadata":   {
                    "headings":     headings,
                    "page_numbers": sorted(set(pages)),
                    "is_table":     is_table,
                },
                "char_count": len(text),
            })

        # Context injection & oversized-chunk splitting
        final_chunks: List[Dict[str, Any]] = []
        n_oversized = 0

        for c in raw_chunks:
            # Inject context breadcrumbs at the top of the chunk
            if c["metadata"].get("headings"):
                breadcrumb  = "[Context: " + " > ".join(c["metadata"]["headings"]) + "]\n"
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
                "re-split to prevent LM overflow."
            )

        print(f"[Agent 1B | Stage 2] Produced {len(final_chunks)} native hierarchical chunks.")
        return final_chunks

    except Exception as exc:
        print(f"[Agent 1B | Stage 2] Native chunking failed ({exc}).")
        return [{
            "chunk_id":   0,
            "content":    "Error extracting document text natively.",
            "metadata":   {"chunking_error": f"{type(exc).__name__}: {exc}"},
            "char_count": 0,
        }]