from typing import Any, Dict, List, Optional, TypedDict

class Agent2BState(TypedDict):
    """Internal execution state for Agent 2B: Rule Extractor."""
    source_file: str

    # Input from Agent 2A (list of dicts with 'chunk_id', 'content', 'entities')
    chunks_with_entities: List[Dict[str, Any]]

    # Output: list of chunk dicts each containing 'extracted_rules' (structured records)
    extracted_rules: Optional[List[Dict[str, Any]]]
    status: str
    error_message: Optional[str]
