from langgraph.graph import END, StateGraph
from .agent_2a_state import Agent2AState
from .agent_2a_tools import extract_entities_from_chunk

def extraction_node(state: Agent2AState) -> dict:
    """Iterates over text chunks and extracts entities synchronously."""
    print(f"\n[Agent 2A] Starting Zero-Shot Entity Extraction for: {state['source_file']}")
    
    chunks = state.get("chunks", [])
    if not chunks:
        return {"status": "error", "error_message": "No chunks provided."}

    extracted_data = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("content", "")
        if not chunk_text.strip():
            continue
            
        print(f"  -> Processing chunk {i+1}/{total_chunks} (ID: {chunk['chunk_id']})...")
        entities = extract_entities_from_chunk(chunk_text)
        
        extracted_data.append({
            "chunk_id": chunk["chunk_id"],
            "metadata": chunk.get("metadata", {}),
            "content": chunk_text,
            "entities": entities
        })
        
    print(f"[Agent 2A] Extraction complete. Processed {total_chunks} chunks.")
    
    return {
        "extracted_entities": extracted_data,
        "status": "complete"
    }

_workflow = StateGraph(Agent2AState)
_workflow.add_node("extract", extraction_node)
_workflow.set_entry_point("extract")
_workflow.add_edge("extract", END)

agent_2a_app = _workflow.compile()