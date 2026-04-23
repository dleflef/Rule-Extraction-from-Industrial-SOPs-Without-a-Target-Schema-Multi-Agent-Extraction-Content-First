from typing import Any, Dict, List, Optional, TypedDict


class Agent2BState(TypedDict):
    """Internal execution state for Agent 2B: Relation Extractor (with Rule Reification)."""
    source_file: str
    prompt_mode: str  # "zero_shot" | "graph_informed"

    # Input from Agent 2A: list of chunk dicts with 'chunk_id', 'content', 'metadata', 'entities'
    chunks_with_entities: List[Dict[str, Any]]

    # Output: list of chunk dicts each containing a 'relations' key
    extracted_relations: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]
