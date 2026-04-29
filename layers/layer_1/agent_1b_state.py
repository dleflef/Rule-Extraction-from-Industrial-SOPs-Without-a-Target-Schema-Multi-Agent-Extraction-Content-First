"""
layer_1/agent_1b_state.py — Agent 1B Internal State

Defines the TypedDict that flows through every node of the Agent 1B
StateGraph (Unstructured Text Parser).

Design rationale
────────────────
Instead of discarding Docling's rich structural metadata by converting
the output to a flat Markdown string, Stage 1 serializes the native
DoclingDocument Pydantic model into a dictionary. Stage 2 reconstructs
this model to perform highly accurate native semantic chunking.
"""

from typing import Any, Dict, List, Optional, TypedDict


class Agent1BState(TypedDict):
    """Internal execution state for Agent 1B: Unstructured Text Parser."""

    # ── Input ────────────────────────────────────────────────────────────────
    file_path: str
    """Absolute or relative path to the unstructured source file."""

    # ── Stage 1 output (parse_node) ──────────────────────────────────────────
    docling_document_dict: Optional[Dict[str, Any]]
    """
    Serialized representation of the DoclingDocument Pydantic datatype.
    Preserves the complete document hierarchy (body, groups, texts) and
    disambiguates the main body from headers/footers (furniture).
    Dropped from state after chunking to free memory.
    """

    page_count: Optional[int]
    """Number of pages Docling detected in the source document."""

    # ── Stage 2 output (chunk_node) ──────────────────────────────────────────
    markdown_chunks: Optional[List[Dict[str, Any]]]
    """
    List of semantic chunk dicts produced by Docling's native HierarchicalChunker.
    Each dict has the form:
        {
            "chunk_id"    : Union[int, str],   # int for original, str for sub-chunks ("2_sub00")
            "content"     : str,               # Markdown text of the chunk
            "metadata"    : {
                "headings"    : ["Chapter 3", "Section 3.1"],
                "page_numbers": [12, 13],       # Provenance for Layer 2
                "source_file" : str,            # Origin file path
                "agent_id"    : "agent_1b",     # Orchestrator construction log
                "processed_at": str,            # ISO-8601 UTC timestamp
            },
            "char_count"  : int,
        }
    """

    chunk_count: Optional[int]
    """Total number of semantic chunks produced by Stage 2."""

    # ── Control flow ─────────────────────────────────────────────────────────
    status: str
    """
    Lightweight state machine for conditional edge routing.
    Transitions: 'pending' → 'parsed' → 'complete' (or 'error')
    """

    error_message: Optional[str]
    """Human-readable failure description, populated when status == 'error'."""