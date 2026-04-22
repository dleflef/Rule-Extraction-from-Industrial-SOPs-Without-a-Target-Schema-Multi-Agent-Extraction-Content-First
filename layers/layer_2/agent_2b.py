from langgraph.graph import END, StateGraph
from .agent_2b_state import Agent2BState
from .agent_2b_tools import extract_relations_from_chunk

def relation_node(state: Agent2BState) -> dict:
    """Iterates over chunks and extracted entities to find relationships."""
    print(f"\n[Agent 2B] Starting Relation Extraction for: {state['source_file']}")
    
    chunks = state.get("chunks_with_entities", [])
    if not chunks:
        return {"status": "error", "error_message": "No chunks provided."}

    extracted_data = []
    total_chunks = len(chunks)     
    
    for i, chunk in enumerate(chunks):
        print(f"  -> Processing chunk {i+1}/{total_chunks} (ID: {chunk.get('chunk_id', 'unknown')})...")
        
        # To get the text, we need the original text content. 
        # Since Agent 2A didn't explicitly save the text in its output array, 
        # we assume it's passed or we match it. We must ensure the text is available.
        chunk_text = chunk.get("content", "")
        entities = chunk.get("entities", [])
        
        if not chunk_text.strip() or not entities:
            # Save empty relations if nothing to process
            extracted_data.append({
                "chunk_id": chunk.get("chunk_id"),
                "metadata": chunk.get("metadata", {}),
                "entities": entities,
                "relations": []
            })
            continue
            
        relations = extract_relations_from_chunk(chunk_text, entities)
        
        extracted_data.append({
            "chunk_id": chunk.get("chunk_id"),
            "metadata": chunk.get("metadata", {}),
            "entities": entities,
            "relations": relations
        })
        
    print(f"[Agent 2B] Extraction complete. Processed {total_chunks} chunks.")
    
    return {
        "extracted_relations": extracted_data,
        "status": "complete"
    }

_workflow = StateGraph(Agent2BState)
_workflow.add_node("extract_relations", relation_node)
_workflow.set_entry_point("extract_relations")
_workflow.add_edge("extract_relations", END)

agent_2b_app = _workflow.compile()