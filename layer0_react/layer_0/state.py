import operator
from typing import TypedDict, Sequence, Annotated, Dict, Any, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SourceItem(TypedDict):
    source_id: str
    file_path: str
    file_type: str
    is_structured: bool
    status: str

class OrchestratorState(TypedDict):
    """State of the Layer 0 Orchestrator for a single execution run."""
    # Conversation memory for the ReAct loop
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Global state variables (overwritten on update)
    inventory: List[SourceItem]
    
    # Scientific evaluation and observability log (appended on update)
    construction_log: Annotated[List[Dict[str, Any]], operator.add]

    # Number of times the guardrail has fired in this run
    guardrail_count: int

    # Number of times evaluate_subordinate_output has been invoked
    validation_invocations: int