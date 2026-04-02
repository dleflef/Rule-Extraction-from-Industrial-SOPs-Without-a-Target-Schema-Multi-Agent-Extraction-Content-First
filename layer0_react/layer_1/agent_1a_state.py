"""
layer_1/agent_1a_state.py — Agent 1A Internal State

Defines the TypedDict that flows through every node of the Agent 1A
StateGraph. This state is strictly focused on Layer 1 ingestion, 
handling schema extraction and basic data standardization.

  Stage 1 (schema_node)          → raw_schema
  Stage 2 (standardization_node) → standardized_data
"""

from typing import TypedDict, Optional, Dict, Any


class Agent1AState(TypedDict):
    """Internal execution state for Agent 1A: Structured Data Ingestor."""

    # ── Input ────────────────────────────────────────────────────────────────
    file_path: str
    """Absolute or relative path to the structured source file (.csv / .json)."""

    # ── Stage 1 output (schema_node) ─────────────────────────────────────────
    raw_schema: Optional[Dict[str, Any]]
    """
    Lightweight schema snapshot produced by programmatic inspection.
    Contains: file_type, columns, dtypes (CSV only), sample_rows, total_columns.
    The full dataset is NEVER loaded at this stage — only headers + 5 rows.
    """

    # ── Stage 2 output (standardization_node) ────────────────────────────────
    standardized_data: Optional[Dict[str, Any]]
    """
    Result of deterministic data standardization executed by Pandas.
    Contains: total_rows, cleaned_columns, and the normalized data records.
    Prepares the data for handoff to Layer 2 (Extraction and Alignment).
    """

    # ── Control flow ─────────────────────────────────────────────────────────
    status: str
    """
    Lightweight state machine for conditional edge routing.
    Valid transitions:
      'pending'          → initial value
      'schema_extracted' → after schema_node succeeds
      'standardized'     → after standardization_node succeeds
      'error'            → any stage failed; error_message is populated
    """

    error_message: Optional[str]
    """Human-readable description of the failure, populated when status=='error'."""