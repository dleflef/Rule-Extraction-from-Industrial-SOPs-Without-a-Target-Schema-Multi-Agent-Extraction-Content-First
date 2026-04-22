from typing import Any, Dict, List, Optional, TypedDict

class Agent2CState(TypedDict):
    """Internal execution state for Agent 2C: Ontology Alignment."""
    source_file: str
    
    # The reference dictionary of official nodes from your Seed Graph
    # Format: {"ST01_FILLING": "Component", "ST01_FILLING_TMP": "Sensor", ...}
    official_nodes: Dict[str, str]
    
    # Input from Agent 2B
    chunks_with_relations: List[Dict[str, Any]]
    
    # Output: Triples that have been aligned to the official KG seed schema
    aligned_relations: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]