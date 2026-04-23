from langgraph.graph import END, StateGraph
from .agent_2a_state import Agent2AState
from .agent_2a_tools import extract_entities_from_chunk, PROMPT_MODE


def extraction_node(state: Agent2AState) -> dict:
    """Iterates over text chunks and extracts entities synchronously."""
    prompt_mode = state.get("prompt_mode") or PROMPT_MODE
    print(f"\n[Agent 2A] Entity Extraction (mode={prompt_mode}) for: {state['source_file']}")

    chunks = state.get("chunks", [])
    if not chunks:
        return {"status": "error", "error_message": "No chunks provided."}

    extracted_data = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("content", "")
        metadata   = chunk.get("metadata", {})
        is_table   = metadata.get("is_table", False)

        if not chunk_text.strip():
            continue

        print(f"  -> Chunk {i+1}/{total_chunks} (ID: {chunk['chunk_id']}, table={is_table})...")
        entities = extract_entities_from_chunk(
            chunk_text,
            prompt_mode=prompt_mode,
            is_table=is_table,
        )

        # Attach provenance to each entity so downstream agents can trace back
        pages = metadata.get("page_numbers", [])
        for ent in entities:
            ent["source_chunk_id"] = chunk["chunk_id"]
            if pages:
                ent["source_page"] = pages[0]

        extracted_data.append({
            "chunk_id": chunk["chunk_id"],
            "metadata": metadata,
            "content":  chunk_text,
            "entities": entities,
        })

    print(f"[Agent 2A] Done. {total_chunks} chunks processed.")

    return {
        "extracted_entities": extracted_data,
        "status": "complete",
    }


_workflow = StateGraph(Agent2AState)
_workflow.add_node("extract", extraction_node)
_workflow.set_entry_point("extract")
_workflow.add_edge("extract", END)

agent_2a_app = _workflow.compile()
