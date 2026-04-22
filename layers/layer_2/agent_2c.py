from langgraph.graph import END, StateGraph
from .agent_2c_state import Agent2CState
from .agent_2c_tools import align_relations_sync

def alignment_node(state: Agent2CState) -> dict:
    """Iterates over chunks and aligns triples to the ontology."""
    print(f"\n[Agent 2C] Starting Ontology Alignment for: {state['source_file']}")
    
    chunks = state.get("chunks_with_relations", [])
    official_nodes = state.get("official_nodes", {})
    
    if not chunks:
        return {"status": "error", "error_message": "No chunks provided."}

    extracted_data = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("content", "")
        raw_relations = chunk.get("relations", [])
        
        if not raw_relations:
            # Pass through empty chunks
            extracted_data.append({
                "chunk_id": chunk.get("chunk_id"),
                "metadata": chunk.get("metadata", {}),
                "aligned_relations": []
            })
            continue
            
        print(f"  -> Aligning chunk {i+1}/{total_chunks} (ID: {chunk.get('chunk_id')})...")
        aligned_relations = align_relations_sync(chunk_text, raw_relations, official_nodes)
        
        extracted_data.append({
            "chunk_id": chunk.get("chunk_id"),
            "metadata": chunk.get("metadata", {}),
            "aligned_relations": aligned_relations
        })
        
    print(f"[Agent 2C] Alignment complete. Processed {total_chunks} chunks.")
    
    return {
        "aligned_relations": extracted_data,
        "status": "complete"
    }

_workflow = StateGraph(Agent2CState)
_workflow.add_node("align_relations", alignment_node)
_workflow.set_entry_point("align_relations")
_workflow.add_edge("align_relations", END)

agent_2c_app = _workflow.compile()