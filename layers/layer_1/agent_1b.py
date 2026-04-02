"""
layer_1/agent_1b.py — Agent 1B: Unstructured Text Parser (LangGraph)
"""

from langgraph.graph import END, StateGraph

from .agent_1b_state import Agent1BState
from .agent_1b_tools import chunk_docling_document, parse_pdf_to_docling


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Document Parser (Docling, Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def parse_node(state: Agent1BState) -> dict:
    """
    Parses the document into a hierarchical Docling Pydantic model.
    """
    print(f"[Agent 1B | parse_node] Parsing: {state['file_path']}")

    result = parse_pdf_to_docling(state["file_path"])

    if "error" in result:
        print(f"[Agent 1B | parse_node] ERROR — {result['error']}")
        return {
            "docling_document_dict": None,
            "page_count":            None,
            "status":                "error",
            "error_message":         result["error"],
        }

    return {
        "docling_document_dict": result["docling_dict"],
        "page_count":            result.get("page_count"),
        "status":                "parsed",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Semantic Native Chunker (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_node(state: Agent1BState) -> dict:
    """
    Splits the DoclingDocument into chunks using its native hierarchy tree.
    """
    print(f"[Agent 1B | chunk_node] Applying native hierarchical chunking.")

    chunks = chunk_docling_document(state["docling_document_dict"])

    if chunks and "chunking_error" in chunks[0].get("metadata", {}):
        err = chunks[0]["metadata"]["chunking_error"]
        print(f"[Agent 1B | chunk_node] ERROR — {err}")
        return {
            "markdown_chunks":       None,
            "chunk_count":           0,
            "status":                "error",
            "error_message":         err,
            "docling_document_dict": None,
        }

    print(f"[Agent 1B | chunk_node] {len(chunks)} chunks produced.")

    return {
        "markdown_chunks": chunks,
        "chunk_count":     len(chunks),
        "status":          "complete",
        # Optional memory optimization: drop the massive dict once chunked
        "docling_document_dict": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge — Error Short-Circuit after parse_node
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_parse(state: Agent1BState) -> str:
    return "end" if state.get("status") == "error" else "chunk"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Compilation
# ─────────────────────────────────────────────────────────────────────────────

_workflow = StateGraph(Agent1BState)

_workflow.add_node("parse", parse_node)
_workflow.add_node("chunk", chunk_node)

_workflow.set_entry_point("parse")

_workflow.add_conditional_edges(
    "parse",
    _route_after_parse,
    {"chunk": "chunk", "end": END},
)

_workflow.add_edge("chunk", END)

agent_1b_app = _workflow.compile()