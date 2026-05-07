"""
layer_1/agent_1b.py — Agent 1B: Unstructured Text Parser (LangGraph)

Converts a document (PDF/DOCX/etc.) into:
  1. markdown_chunks  — in-memory list used by downstream Layer 1 agents.
  2. output_txt_path  — a plain .txt file written to texts_dir (default "texts").

The .txt file is the primary feed for step2_grid_search_extraction.py, which
reads every .txt in its texts/ directory and passes each to an LLM for rule
extraction. Chunk content is joined with double newlines so section structure
is preserved in the file.
"""

from langgraph.graph import END, StateGraph

from .agent_1b_state import Agent1BState
from .agent_1b_tools import convert_document_to_sop_txt, parse_pdf_to_markdown_chunks


def parse_node(state: Agent1BState) -> dict:
    """
    Parse the document, populate markdown_chunks, and save a .txt to texts_dir.
    On failure the state transitions to 'error'.
    """
    file_path = state["file_path"]
    texts_dir = state.get("texts_dir", "texts")
    print(f"[Agent 1B] Parsing: {file_path} → {texts_dir}/")

    try:
        chunks = parse_pdf_to_markdown_chunks(file_path)
        # Reuse the already-parsed chunks so Docling runs only once.
        txt_path = convert_document_to_sop_txt(file_path, texts_dir, chunks=chunks)
        print(
            f"[Agent 1B] Success — {len(chunks)} chunks extracted, "
            f"saved to {txt_path}"
        )
        return {
            "markdown_chunks": chunks,
            "chunk_count": len(chunks),
            "output_txt_path": txt_path,
            "status": "complete",
        }
    except Exception as exc:
        print(f"[Agent 1B] ERROR: {exc}")
        return {
            "markdown_chunks": None,
            "chunk_count": 0,
            "output_txt_path": None,
            "status": "error",
            "error_message": str(exc),
        }


# ── Graph compilation ──────────────────────────────────────────────────────
_workflow = StateGraph(Agent1BState)
_workflow.add_node("parse", parse_node)
_workflow.set_entry_point("parse")
_workflow.add_edge("parse", END)

agent_1b_app = _workflow.compile()