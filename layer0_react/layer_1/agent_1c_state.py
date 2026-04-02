"""
layer_1/agent_1c_state.py — Agent 1C Internal State

Defines the TypedDict that flows through every node of the Agent 1C
StateGraph (Sensor & Signal Mapper). This state is strictly focused on 
Layer 1 ingestion, handling programmatic log sampling, LLM-assisted 
regex generation for structural parsing, and data standardization.
"""

from typing import Any, Dict, List, Optional, TypedDict


class Agent1CState(TypedDict):
    """Internal execution state for Agent 1C: Sensor & Signal Mapper."""
    
    # ── Input ────────────────────────────────────────────────────────────────
    file_path: str
    """Absolute or relative path to the unstructured log or signal file."""
    
    # ── Stage 1 output (sampler_node) ────────────────────────────────────────
    sample_lines: Optional[List[str]]
    """A small peek (e.g., 20 lines) into the log file to minimize LLM token usage."""
    
    # ── Stage 2 output (regex_generator_node) ────────────────────────────────
    structural_regex: Optional[str]
    """
    LLM-generated regex pattern with generic named capture groups representing
    the structural columns of the log (e.g., timestamp, log_level, module).
    No ontology terms are applied here.
    """
    
    # ── Stage 3 output (standardization_node) ────────────────────────────────
    standardized_logs: Optional[Dict[str, Any]]
    """
    The bulk-parsed records formatted as a standardized list of dictionaries,
    ready for handoff to Layer 2 (Extraction and Alignment).
    Contains: total_lines, parsed_count, unparsed_count, and the normalized data.
    """
    
    # ── Control flow ─────────────────────────────────────────────────────────
    status: str
    """
    Lightweight state machine for conditional edge routing.
    Transitions: 'pending' → 'sampled' → 'regex_generated' → 'standardized' (or 'error')
    """
    
    error_message: Optional[str]
    """Human-readable description of the failure, populated when status=='error'."""