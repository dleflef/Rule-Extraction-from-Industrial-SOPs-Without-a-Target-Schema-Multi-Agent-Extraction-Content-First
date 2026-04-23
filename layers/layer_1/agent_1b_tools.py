"""
layer_1/agent_1b_tools.py — Agent 1B Deterministic Tools

Implements the two purely programmatic tools that form the Agent 1B
pipeline, leveraging Docling's native structure and chunking capabilities.
"""

import os
import re
from typing import Any, Dict, List

_MAX_CHUNK_CHARS: int = 2_500
_CHUNK_OVERLAP:   int = 100


def _sanitize_text(text: str) -> str:
    # Fix table concatenation artifact
    text = text.replace("WARN_LONominal", "WARN_LO Nominal")
    # Restore workflow arrows from ligature corruption variants
    text = re.sub(r"\s›\s*fi\s", " -> ", text)
    text = re.sub(r"fl\s", "-> ", text)
    text = re.sub(r"(?<!\S)fi(?!\S)", "->", text)
    return text


def _split_oversized_chunk(chunk: Dict[str, Any], base_idx: int) -> List[Dict[str, Any]]:
    text     = chunk["content"]
    metadata = chunk["metadata"]
    results  = []
    start    = 0
    sub_idx  = 0

    while start < len(text):
        end  = min(start + _MAX_CHUNK_CHARS, len(text))
        part = text[start:end]
        results.append({
            "chunk_id":   int(f"{base_idx}{sub_idx:02d}"),
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
# F1 & F2 Patch Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_table_markdown_map(doc: Any) -> Dict[str, str]:
    """F2: Maps Docling table references to clean Markdown dataframes."""
    table_map = {}
    if hasattr(doc, "tables"):
        for table in doc.tables:
            try:
                # Convert the table to a pandas DataFrame, then to Markdown
                df = table.export_to_dataframe()
                ref = table.get_ref().dict()["$ref"]
                table_map[ref] = df.to_markdown(index=False)
            except Exception:
                pass
    return table_map


def _promote_list_item_headings(raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """F1: Detects misparsed heading-only chunks and promotes them to metadata."""
    heading_regex = re.compile(r"^-\s+(\d+(?:\.\d+)+)\s+(.+)$")
    cleaned_chunks = []
    
    current_promoted_heading = None

    for chunk in raw_chunks:
        lines = chunk["content"].strip().split('\n')
        # Check if the chunk's body is exclusively a heading pattern
        if len(lines) == 1 and heading_regex.match(lines[0]):
            match = heading_regex.match(lines[0])
            current_promoted_heading = f"{match.group(1)} {match.group(2)}"
            # Drop this chunk (do not append to cleaned_chunks)
            continue
            
        # If we have a promoted heading, apply it to the innermost headings list
        if current_promoted_heading:
            if not chunk["metadata"].get("headings"):
                chunk["metadata"]["headings"] = [current_promoted_heading]
            else:
                chunk["metadata"]["headings"][-1] = current_promoted_heading
                
        cleaned_chunks.append(chunk)

    return cleaned_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Document Parser (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf_to_docling(file_path: str) -> Dict[str, Any]:
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
            max_file_size=100 * 1024 * 1024 
        )

        acceptable_statuses = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}
        if result.status not in acceptable_statuses:
            errors = "; ".join([e.error_message for e in result.errors]) if result.errors else "Unknown"
            return {"error": f"Conversion failed with status '{result.status}'. Errors: {errors}"}

        doc_dict = result.document.model_dump()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else None

        print(f"[Agent 1B | Stage 1] Conversion complete. Pages: {page_count}")
        return {
            "docling_dict": doc_dict,
            "page_count": page_count,
        }

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Semantic Native Chunker (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_docling_document(docling_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not docling_dict:
        return []

    try:
        from docling_core.types.doc import DoclingDocument
        from docling.chunking import HierarchicalChunker

        doc = DoclingDocument.model_validate(docling_dict)
        chunker = HierarchicalChunker()
        doc_chunks = list(chunker.chunk(doc))
        
        # F2: Pre-build the markdown tables
        table_markdown_map = _build_table_markdown_map(doc)

        raw_chunks = []
        for idx, chunk in enumerate(doc_chunks):
            is_table = False
            text = ""
            
            # F2: Intercept tables and replace with our clean Markdown
            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    ref = getattr(item, "get_ref", lambda: None)()
                    if ref and ref.dict().get("$ref") in table_markdown_map:
                        is_table = True
                        text = table_markdown_map[ref.dict()["$ref"]]
                        break
            
            # Fallback to standard text if not a table
            if not is_table:
                text = _sanitize_text(chunk.text.strip()) if hasattr(chunk, "text") else ""
            
            if not text:
                continue

            headings = chunk.meta.headings if hasattr(chunk.meta, "headings") else []
            pages = []
            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        pages.extend([p.page_no for p in item.prov if hasattr(p, "page_no")])

            raw_chunks.append({
                "chunk_id":   idx,
                "content":    text,
                "metadata":   {
                    "headings": headings,
                    "page_numbers": sorted(list(set(pages))),
                    "is_table": is_table
                },
                "char_count": len(text),
            })

        # F1: Promote orphaned list items to headings
        cleaned_chunks = _promote_list_item_headings(raw_chunks)

        # Context Injection & Sizing
        final_chunks: List[Dict[str, Any]] = []
        n_oversized = 0
        
        for c in cleaned_chunks:
            # Prepend breadcrumbs AFTER F1 heading promotion has run
            if c["metadata"].get("headings"):
                breadcrumb = "[Context: " + " > ".join(c["metadata"]["headings"]) + "]\n"
                c["content"] = breadcrumb + c["content"]
                c["char_count"] = len(c["content"])

            if c["char_count"] > _MAX_CHUNK_CHARS:
                n_oversized += 1
                final_chunks.extend(_split_oversized_chunk(c, c["chunk_id"]))
            else:
                final_chunks.append(c)

        if n_oversized:
            print(f"[Agent 1B | Stage 2] {n_oversized} oversized chunk(s) re-split to prevent LM overflow.")
        
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