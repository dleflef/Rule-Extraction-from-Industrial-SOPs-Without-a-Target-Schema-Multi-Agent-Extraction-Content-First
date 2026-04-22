from typing import Any, Dict, List, Optional, TypedDict

class Agent2BState(TypedDict):
    """Internal execution state for Agent 2B: Relation Extractor."""
    source_file: str
    
    # Input from Agent 2A (List of dicts containing 'chunk_id', 'content', and 'entities')
    chunks_with_entities: List[Dict[str, Any]]
    
    # Output: List of chunk mappings with their extracted relation triples
    extracted_relations: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]