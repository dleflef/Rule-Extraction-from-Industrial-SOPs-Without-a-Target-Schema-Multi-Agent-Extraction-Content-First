from typing import Any, Dict, List, Optional, TypedDict

class Agent2CState(TypedDict):
    """Internal execution state for Agent 2C: Ontology Alignment Agent."""
    source_file: str

    # Reference dictionary of official KG seed nodes: {"ST01_FILLING": "Component", ...}
    official_nodes: Dict[str, str]

    # Input from Agent 2B: list of chunk dicts each with a 'relations' key
    chunks_with_relations: List[Dict[str, Any]]

    # Output: same chunk structure with aligned and structurally verified triples
    aligned_relations: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]