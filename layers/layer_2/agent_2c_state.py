from typing import Any, Dict, List, Optional, TypedDict


class Agent2CState(TypedDict):
    """Internal execution state for Agent 2C: Ontology Alignment Agent."""
    source_file: str

    # Reference dictionary of official KG seed nodes: {"ST01_FILLING": "Component", ...}
    official_nodes: Dict[str, str]

    # Input from Agent 2B: list of chunk dicts with 'extracted_rules'
    chunks_with_rules: List[Dict[str, Any]]

    # Output 1: same chunk structure with 'aligned_rules' (station/sensor normalised)
    aligned_rule_chunks: Optional[List[Dict[str, Any]]]

    # Output 2: flat list of KG triples with full properties
    kg_triples: Optional[List[Dict[str, Any]]]

    status: str
    error_message: Optional[str]
