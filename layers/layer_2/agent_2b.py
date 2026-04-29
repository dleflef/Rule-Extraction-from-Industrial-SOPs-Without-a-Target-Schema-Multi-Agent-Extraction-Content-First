from langgraph.graph import END, StateGraph
from .agent_2b_state import Agent2BState
from .agent_2b_tools import extract_relations_from_chunk

def relation_node(state: Agent2BState) -> dict:
    """Iterates over chunks and links Agent 2A's entities into triples."""
    print(f"\n[Agent 2B] Starting Relation Extraction for: {state['source_file']}")

    chunks = state.get("chunks_with_entities", [])
    if not chunks:
        return {"status": "error", "error_message": "No chunks provided to Agent 2B."}

    extracted_data = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        chunk_text = chunk.get("content", "")
        entities = chunk.get("entities", [])
        metadata = chunk.get("metadata", {})

        if not chunk_text.strip() or not entities:
            # Pass through chunks that have no text or no entities found by 2A
            extracted_data.append({
                "chunk_id": chunk_id,
                "metadata": metadata,
                "entities": entities,
                "relations": []
            })
            continue

        print(f"  -> Processing chunk {i+1}/{total_chunks} (ID: {chunk_id})...")
        
        relations = extract_relations_from_chunk(
            chunk_text=chunk_text, 
            entities=entities,
            chunk_id=chunk_id
        )

        extracted_data.append({
            "chunk_id": chunk_id,
            "metadata": metadata,
            "entities": entities,
            "relations": relations
        })

    total_triples = sum(len(c["relations"]) for c in extracted_data)
    print(f"[Agent 2B] Extraction complete. {total_triples} triples from {total_chunks} chunks.")

    return {
        "extracted_relations": extracted_data,
        "status": "complete"
    }

# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------
_workflow = StateGraph(Agent2BState)
_workflow.add_node("extract_relations", relation_node)
_workflow.set_entry_point("extract_relations")
_workflow.add_edge("extract_relations", END)

agent_2b_app = _workflow.compile()