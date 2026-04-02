"""
layer_1/agent_1b_tools.py — Agent 1B Deterministic Tools

Implements the two purely programmatic tools that form the Agent 1B
pipeline, leveraging Docling's native structure and chunking capabilities.
"""

import os
from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Document Parser (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf_to_docling(file_path: str) -> Dict[str, Any]:
    """
    Stage 1 (Docling Vision Parsing).

    Converts a document into a DoclingDocument using the DocumentConverter.
    Enforces safe resource limits (max_num_pages, max_file_size) to protect 
    the pipeline from massive out-of-memory errors on large technical manuals.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    _SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".txt"}
    if ext not in _SUPPORTED:
        return {
            "error": f"Format '{ext}' not supported by Agent 1B. Supported: {sorted(_SUPPORTED)}."
        }

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import ConversionStatus
        
        try:
            from docling.exceptions import ConversionError
        except ImportError:
            ConversionError = RuntimeError

        print(f"[Agent 1B | Stage 1] Starting Docling conversion: {file_path}")
        
        # Initialize the converter based on format preferences
        converter = DocumentConverter()
        
        # Convert document with built-in limits to protect hardware resources
        # 500 pages and 100MB are reasonable safety constraints for academic/industrial PDFs
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

        # Serialize the Pydantic DoclingDocument to a dict for LangGraph State
        doc_dict = result.document.model_dump()
        
        # Safely extract page count
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
    """
    Stage 2 (Native Semantic Chunking).

    Reconstructs the DoclingDocument from state and uses Docling's native 
    HierarchicalChunker. This ensures chunk boundaries perfectly map to the 
    document's internal JSON-pointer tree, keeping groups and list elements together.
    """
    if not docling_dict:
        return []

    try:
        # Import the core document model and the chunker
        from docling_core.types.doc import DoclingDocument
        from docling.chunking import HierarchicalChunker

        # Rehydrate the Pydantic model from the state dictionary
        doc = DoclingDocument.model_validate(docling_dict)
        
        # Apply the native Hierarchical Chunker
        chunker = HierarchicalChunker()
        doc_chunks = list(chunker.chunk(doc))

        chunks = []
        for idx, chunk in enumerate(doc_chunks):
            # Extract clean text and metadata safely
            text = chunk.text.strip() if hasattr(chunk, "text") else ""
            if not text:
                continue
                
            # Grab heading paths (e.g., ["Chapter 1", "Section 1.2"])
            headings = chunk.meta.headings if hasattr(chunk.meta, "headings") else []
            
            # Grab provenance (page numbers where this chunk appears)
            pages = []
            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        pages.extend([p.page_no for p in item.prov if hasattr(p, "page_no")])

            chunks.append({
                "chunk_id":   idx,
                "content":    text,
                "metadata":   {
                    "headings": headings,
                    "page_numbers": sorted(list(set(pages)))
                },
                "char_count": len(text),
            })

        print(f"[Agent 1B | Stage 2] Produced {len(chunks)} native hierarchical chunks.")
        return chunks

    except Exception as exc:
        print(f"[Agent 1B | Stage 2] Native chunking failed ({exc}).")
        return [{
            "chunk_id":   0,
            "content":    "Error extracting document text natively.",
            "metadata":   {"chunking_error": f"{type(exc).__name__}: {exc}"},
            "char_count": 0,
        }]