import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from utils import safe_preview_file

@tool
def inspect_source(file_path: str) -> str:
    """Use this to safely read the first few lines of a file to understand its schema or content."""
    return safe_preview_file(file_path)

@tool
def route_to_layer_1(source_id: str, agent_type: str, priority: int) -> str:
    """
    Routes a source to a Layer 1 ingestion agent.
    agent_type should be 'structured_parser', 'unstructured_parser', or 'signal_parser'.
    priority should be 1 (High/Structured) or 2 (Low/Unstructured).
    """
    return f"SUCCESS: {source_id} routed to {agent_type} with priority {priority}."

@tool
def evaluate_subordinate_output(source_id: str) -> str:
    """
    Agentic Evaluation: Uses a Validation Agent to check the ingestion output from a Layer 1 agent.
    Layer 1 is responsible for parsing and structuring raw source files — not entity extraction.
    Call this after routing a source to verify that ingestion completed correctly.
    """
    # TODO: replace with a real call to Layer 3 once implemented
    validation_agent = ChatOpenAI(
        model="qwen/qwen3-vl-4b",
        temperature=0.0,
        api_key="lm-studio-local",
        base_url="http://127.0.0.1:1234/v1",
    )

    # Mock of what a Layer 1 Ingestion Agent would actually produce:
    # parsed structure, schema, row/page counts — NOT extracted entities (that is Layer 2's job).
    mock_layer_1_output = (
        f"Ingestion report for {source_id}: "
        f"file parsed successfully, schema detected: [timestamp, sensor_id, value, unit], "
        f"record count: 142, no encoding errors, output format: structured JSON."
    )

    evaluation_prompt = f"""
    You are a Validation Agent in a Knowledge Graph construction pipeline.
    Review the following ingestion report produced by a Layer 1 Ingestion Agent for source {source_id}.
    Layer 1 is responsible for parsing raw files and producing structured, clean output ready for
    downstream entity extraction (Layer 2). It does NOT extract entities itself.

    Ingestion report to evaluate:
    {mock_layer_1_output}

    Evaluate whether the ingestion is complete and the output is structurally sound:
    - Was the file parsed without errors?
    - Is the schema or structure identifiable?
    - Is the record/page count plausible?
    - Is the output format suitable for Layer 2 to consume?

    Return a valid JSON object EXACTLY matching this schema:
    {{
      "status": "success" or "failed",
      "confidence_score": float (0.0 to 1.0, representing your confidence in ingestion quality),
      "reason": "brief explanation of your evaluation"
    }}
    """
    
    try:
        response = validation_agent.invoke(evaluation_prompt)
        # Clean response to ensure it's pure JSON
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return clean_json
    except Exception as e:
        return json.dumps({
            "status": "failed", 
            "confidence_score": 0.0, 
            "reason": f"Validation Agent stalled. Error: {str(e)}"
        })

@tool
def escalate_issue(source_id: str, reason: str) -> str:
    """Escalates a source that repeatedly fails evaluation."""
    return f"ESCALATED: {source_id}. Reason logged: {reason}."

# Map tools for the LangGraph node execution
tools = [inspect_source, route_to_layer_1, evaluate_subordinate_output, escalate_issue]
tools_by_name = {t.name: t for t in tools}