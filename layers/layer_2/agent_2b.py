from langgraph.graph import END, StateGraph
from .agent_2b_state import Agent2BState
from .agent_2b_tools import extract_rules_from_chunk


def rule_extraction_node(state: Agent2BState) -> dict:
    """Iterates over chunks and extracts structured rule records matching the GT schema."""
    print(f"\n[Agent 2B] Starting Rule Extraction for: {state['source_file']}")

    chunks = state.get("chunks_with_entities", [])
    if not chunks:
        return {"status": "error", "error_message": "No chunks provided."}

    extracted_data = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        print(f"  -> Processing chunk {i+1}/{total_chunks} (ID: {chunk.get('chunk_id', 'unknown')})...")

        chunk_text = chunk.get("content", "")

        if not chunk_text.strip():
            extracted_data.append({
                "chunk_id": chunk.get("chunk_id"),
                "metadata": chunk.get("metadata", {}),
                "entities": chunk.get("entities", []),
                "extracted_rules": []
            })
            continue

        rules = extract_rules_from_chunk(chunk_text)

        extracted_data.append({
            "chunk_id": chunk.get("chunk_id"),
            "metadata": chunk.get("metadata", {}),
            "entities": chunk.get("entities", []),
            "extracted_rules": rules
        })

    total_rules = sum(len(c["extracted_rules"]) for c in extracted_data)
    print(f"[Agent 2B] Extraction complete. {total_rules} rules extracted from {total_chunks} chunks.")

    return {
        "extracted_rules": extracted_data,
        "status": "complete"
    }


_workflow = StateGraph(Agent2BState)
_workflow.add_node("extract_rules", rule_extraction_node)
_workflow.set_entry_point("extract_rules")
_workflow.add_edge("extract_rules", END)

agent_2b_app = _workflow.compile()
