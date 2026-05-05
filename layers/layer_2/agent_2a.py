"""
layer_2/agent_2a.py — Agent 2A: Entity Extractor (LangGraph)

Processes one chunk at a time. The Orchestrator iterates over chunks,
calls this agent, and aggregates the extracted rules.
"""

from langgraph.graph import END, StateGraph

from .agent_2a_state import Agent2AState
from .agent_2a_tools import extract_rules_from_chunk


def extract_node(state: Agent2AState) -> dict:
    """
    Runs entity extraction on a single chunk.
    """
    chunk = state["chunk"]
    content = chunk.get("content", "")
    headings = chunk.get("metadata", {}).get("headings", [])

    seed_path = state.get("seed_nodes_csv_path", "layers/data/seed_rules/dataset/kg_seeds/nodes_factory.csv")
    model = "qwen2.5-coder-7b-instruct"

    print(f"[Agent 2A] Extracting rules from chunk {chunk.get('chunk_id')} "
          f"(head: {headings})")

    rules = extract_rules_from_chunk(
        chunk_content=content,
        headings=headings,
        seed_nodes_csv=seed_path,
        model_name=model,
        temperature=0.0,
    )

    print(f"[Agent 2A] Extracted {len(rules)} rule(s).")
    return {
        "extracted_rules": rules,
        "extraction_status": "complete",
        "extraction_model": model,
    }


# ── Graph compilation (single node) ──────────────────────────────────────────
_workflow = StateGraph(Agent2AState)
_workflow.add_node("extract", extract_node)
_workflow.set_entry_point("extract")
_workflow.add_edge("extract", END)

agent_2a_app = _workflow.compile()