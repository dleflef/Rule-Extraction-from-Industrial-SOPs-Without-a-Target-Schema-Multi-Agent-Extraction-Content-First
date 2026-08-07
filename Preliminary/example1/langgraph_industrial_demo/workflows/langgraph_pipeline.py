from langgraph.graph import StateGraph, START, END
from workflows.state_schema import IndustrialState
from agents.all_agents import (
    ingestion_agent, sensor_agent, iot_agent, 
    log_agent, ocr_agent, analysis_agent, neo4j_agent
)

def should_continue(state: IndustrialState):
    # Check if files were successfully extracted.
    paths = state.get("file_paths", {})
    if any(paths.values()):
        return "continue"
    return "end"

def dispatch_agent(state: dict, **kwargs) -> dict:
    # A tiny pass-through node to handle the parallel Fan-out safely
    return {}

def build_workflow():
    workflow = StateGraph(IndustrialState)

    # Register all nodes
    workflow.add_node("Ingestion_Node", ingestion_agent)
    workflow.add_node("Dispatcher_Node", dispatch_agent) 
    workflow.add_node("Sensor_Node", sensor_agent)
    workflow.add_node("IoT_Node", iot_agent)
    workflow.add_node("Log_Node", log_agent)
    workflow.add_node("OCR_Node", ocr_agent)
    workflow.add_node("Neo4j_Node", neo4j_agent)
    workflow.add_node("Analysis_Node", analysis_agent)

    # Start the pipeline
    workflow.add_edge(START, "Ingestion_Node")
    
    # Conditional route to the Dispatcher or abort
    workflow.add_conditional_edges(
        "Ingestion_Node",
        should_continue,
        {
            "continue": "Dispatcher_Node",
            "end": END
        }
    )
    
    # Safely Fan-out from the Dispatcher (Parallel Processing)
    workflow.add_edge("Dispatcher_Node", "Sensor_Node")
    workflow.add_edge("Dispatcher_Node", "IoT_Node")
    workflow.add_edge("Dispatcher_Node", "Log_Node")
    workflow.add_edge("Dispatcher_Node", "OCR_Node")
    
    # Fan-in to Neo4j Node
    workflow.add_edge("Sensor_Node", "Neo4j_Node")
    workflow.add_edge("IoT_Node", "Neo4j_Node")
    workflow.add_edge("Log_Node", "Neo4j_Node")
    
    # Final consolidation for Analysis
    workflow.add_edge("Neo4j_Node", "Analysis_Node")
    workflow.add_edge("OCR_Node", "Analysis_Node")
    
    # End
    workflow.add_edge("Analysis_Node", END)

    return workflow.compile()
