"""
layer_1/agent_1b.py — Agent 1B: Unstructured Text Parser (LangGraph)

Uses a single parse node to convert a document into semantic chunks.
The graph is intentionally minimal — the real work lives in agent_1b_tools.py.
"""

from langgraph.graph import END, StateGraph

from .agent_1b_state import Agent1BState
from .agent_1b_tools import parse_pdf_to_markdown_chunks


def parse_node(state: Agent1BState) -> dict:
    """
    Calls the document parser and populates markdown_chunks.
    On failure the state transitions to 'error'.
    """
    print(f"[Agent 1B] Parsing: {state['file_path']}")

    try:
        chunks = parse_pdf_to_markdown_chunks(state["file_path"])
        print(f"[Agent 1B] Success — {len(chunks)} chunks extracted.")
        return {
            "markdown_chunks": chunks,
            "chunk_count": len(chunks),
            "status": "complete",
        }
    except Exception as exc:
        print(f"[Agent 1B] ERROR: {exc}")
        return {
            "markdown_chunks": None,
            "chunk_count": 0,
            "status": "error",
            "error_message": str(exc),
        }


# ── Graph compilation ──────────────────────────────────────────────────────
_workflow = StateGraph(Agent1BState)
_workflow.add_node("parse", parse_node)
_workflow.set_entry_point("parse")
_workflow.add_edge("parse", END)

agent_1b_app = _workflow.compile()