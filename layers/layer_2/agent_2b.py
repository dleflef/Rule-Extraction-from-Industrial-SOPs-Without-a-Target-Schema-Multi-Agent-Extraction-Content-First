from langgraph.graph import END, StateGraph
from .agent_2b_state import Agent2BState
from .agent_2b_tools import extract_relations_from_chunk, PROMPT_MODE


def relation_node(state: Agent2BState) -> dict:
    """
    Iterates over chunks and extracts reified rule triples.
    Each rule is represented as a Rule node hub with typed attribute triples.
    """
    prompt_mode = state.get("prompt_mode") or PROMPT_MODE
    print(f"\n[Agent 2B] Starting Relation Extraction (mode={prompt_mode}) for: {state['source_file']}")

    chunks = state.get("chunks_with_entities", [])
    if not chunks:
        return {"status": "error", "error_message": "No chunks provided."}

    extracted_data = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_id  = chunk.get("chunk_id", f"chunk_{i}")
        print(f"  -> Processing chunk {i+1}/{total_chunks} (ID: {chunk_id})...")

        chunk_text = chunk.get("content", "")
        entities   = chunk.get("entities", [])
        metadata   = chunk.get("metadata", {})
        is_table   = metadata.get("is_table", False)

        if not chunk_text.strip():
            extracted_data.append({
                "chunk_id": chunk_id,
                "metadata": metadata,
                "entities": entities,
                "relations": []
            })
            continue

        relations = extract_relations_from_chunk(
            chunk_text, entities,
            prompt_mode=prompt_mode,
            is_table=is_table,
            chunk_id=chunk_id,
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


_workflow = StateGraph(Agent2BState)
_workflow.add_node("extract_relations", relation_node)
_workflow.set_entry_point("extract_relations")
_workflow.add_edge("extract_relations", END)

agent_2b_app = _workflow.compile()
