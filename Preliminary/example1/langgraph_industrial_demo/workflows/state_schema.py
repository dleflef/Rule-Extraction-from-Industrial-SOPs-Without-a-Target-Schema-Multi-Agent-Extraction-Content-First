from typing import TypedDict, Dict, Any, List

# Common dictionary for the state used across all agents in the LangGraph pipeline.
class IndustrialState(TypedDict):
    file_paths: Dict[str, str]
    
    sensor_summary: Dict[str, Any] 
    iot_payloads: List[Dict[str, Any]]
    parsed_logs: List[str]

    ocr_result: Dict[str, Any] 
    final_report: str
    neo4j_status: str
