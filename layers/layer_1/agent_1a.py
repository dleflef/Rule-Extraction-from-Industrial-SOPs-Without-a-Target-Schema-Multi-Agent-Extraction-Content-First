"""
layer_1/agent_1a.py — Agent 1A: Structured Data Ingestor (LangGraph)

Implements the Layer 1 Ingestion pipeline for structured data.
This agent is strictly programmatic and deterministic, acting as
the data formatting and ingestion layer before handing off to
Layer 2's reasoning agents.

Architecture (two sequential nodes)
──────────────────────────────────────

  ┌──────────────┐     ┌──────────────────────┐
  │  schema_node │────▶│ standardization_node │
  │  (Stage 1)   │     │ (Stage 2)            │
  │              │     │                      │
  │  Programmatic│     │  Programmatic        │
  │  5-row peek  │     │  Pandas full-run     │
  └──────────────┘     └──────────────────────┘
         │                        │ 
         └───────── END ──────────┘
"""

from langgraph.graph import END, StateGraph

from .agent_1a_state import Agent1AState
from .agent_1a_tools import extract_schema, standardize_data


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Schema Extractor (Programmatic, Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def schema_node(state: Agent1AState) -> dict:
    """
    Reads column headers and 5 sample rows from the source file.

    State transitions:
      pending → schema_extracted  (success)
      pending → error             (file not found / unsupported type)
    """
    print(f"[Agent 1A | Stage 1] Extracting schema from: {state['file_path']}")

    raw_schema = extract_schema(state["file_path"])

    if "error" in raw_schema:
        print(f"[Agent 1A | Stage 1] ERROR: {raw_schema['error']}")
        return {
            "raw_schema": raw_schema,
            "status": "error",
            "error_message": raw_schema["error"],
        }

    print(
        f"[Agent 1A | Stage 1] Schema extracted: "
        f"{raw_schema['total_columns']} columns, "
        f"type={raw_schema['file_type']}"
    )
    return {"raw_schema": raw_schema, "status": "schema_extracted"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Data Standardizer (Programmatic, Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def standardization_node(state: Agent1AState) -> dict:
    """
    Standardizes the full dataset into a normalized format.
    
    State transitions:
      schema_extracted → complete  (Pandas processing succeeded)
      schema_extracted → error         (I/O or structural failure)
    """
    print(
        f"[Agent 1A | Stage 2] Executing data standardization on: "
        f"{state['file_path']}"
    )

    result = standardize_data(state["file_path"])

    if result.get("status") != "success":
        err = result.get("error", result.get("error_message", "standardize_data did not return status='success'"))
        print(f"[Agent 1A | Stage 2] ERROR: {err}")
        return {
            "standardized_data": None,
            "status": "error",
            "error_message": err,
        }

    print(
        f"[Agent 1A | Stage 2] Standardization complete — "
        f"{result['total_rows']} rows prepared for Layer 2."
    )

    return {"standardized_data": result, "status": "standardized"}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge — Error Short-Circuit
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_schema(state: Agent1AState) -> str:
    """
    After schema_node, decide whether to proceed or short-circuit to END.
    """
    if state.get("status") == "error":
        return "end"
    return "standardize"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Compilation
# ─────────────────────────────────────────────────────────────────────────────

_workflow = StateGraph(Agent1AState)

_workflow.add_node("schema", schema_node)
_workflow.add_node("standardize", standardization_node)

_workflow.set_entry_point("schema")

# schema → standardize (happy path)
# schema → END         (error short-circuit)
_workflow.add_conditional_edges(
    "schema",
    _route_after_schema,
    {"standardize": "standardize", "end": END},
)

# standardize is always the terminal node on the happy path
_workflow.add_edge("standardize", END)

# Public handle
agent_1a_app = _workflow.compile()