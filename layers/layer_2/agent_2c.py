from langgraph.graph import END, StateGraph
from .agent_2c_state import Agent2CState
from .agent_2c_tools import align_and_convert


def alignment_node(state: Agent2CState) -> dict:
    """
    Aligns station/sensor names in structured rule records to official KG seed
    node IDs, then converts each rule to KG triples with full properties.
    """
    print(f"\n[Agent 2C] Starting Ontology Alignment for: {state['source_file']}")

    chunks = state.get("chunks_with_rules", [])
    official_nodes = state.get("official_nodes", {})

    if not chunks:
        return {"status": "error", "error_message": "No chunks_with_rules provided."}

    aligned_rule_chunks, kg_triples = align_and_convert(chunks, official_nodes)

    total_rules = sum(len(c["aligned_rules"]) for c in aligned_rule_chunks)
    print(
        f"[Agent 2C] Alignment complete. "
        f"{total_rules} rules aligned, {len(kg_triples)} KG triples produced."
    )

    return {
        "aligned_rule_chunks": aligned_rule_chunks,
        "kg_triples": kg_triples,
        "status": "complete"
    }


_workflow = StateGraph(Agent2CState)
_workflow.add_node("align_rules", alignment_node)
_workflow.set_entry_point("align_rules")
_workflow.add_edge("align_rules", END)

agent_2c_app = _workflow.compile()
