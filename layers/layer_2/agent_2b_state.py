from typing import Any, Dict, List, Optional, TypedDict

class Agent2BState(TypedDict):
    """Internal execution state for Agent 2B: Relation Extractor."""
    source_file: str
    
    # Input from Agent 2A: list of chunk dicts with 'chunk_id', 'content', 'metadata', 'entities'
    chunks_with_entities: List[Dict[str, Any]]

    # Output: list of chunk dicts each containing a 'relations' list of triples
    extracted_relations: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]