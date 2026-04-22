from typing import Any, Dict, List, Optional, TypedDict

class Agent2AState(TypedDict):
    """Internal execution state for Agent 2A: Entity Extractor."""
    source_file: str
    chunks: List[Dict[str, Any]]
    
    # List of chunk mappings with their extracted entities
    extracted_entities: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]