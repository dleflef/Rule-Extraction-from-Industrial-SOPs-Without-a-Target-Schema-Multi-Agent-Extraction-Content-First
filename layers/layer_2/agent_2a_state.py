"""
layer_2/agent_2a_state.py — Agent 2A Internal State

Describes the input and output of the Entity Extractor.
"""

from typing import Any, Dict, List, Optional, TypedDict


class Agent2AState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    chunk: Dict[str, Any]
    """
    A single chunk dict produced by Agent 1B:
        {
            "chunk_id" : int,
            "content"  : str,            # Markdown section text
            "metadata" : {
                "headings"     : list[str],
                "page_numbers" : list[int],
            },
        }
    """

    seed_nodes_csv_path: Optional[str]
    """Path to nodes_factory.csv. If not provided, a default is used."""

    # ── Output ───────────────────────────────────────────────────────────────
    extracted_rules: Optional[List[Dict[str, Any]]]
    """
    Each rule dict contains the 14 ground‑truth fields:
        ruleId, class, station, sensor, sensorType, condition, action,
        severity, critHi, warnHi, critLo, warnLo, unit, source
    """

    extraction_status: str            # "pending" → "complete" or "error"
    extraction_error: Optional[str]
    extraction_model: Optional[str]   # record which model was used