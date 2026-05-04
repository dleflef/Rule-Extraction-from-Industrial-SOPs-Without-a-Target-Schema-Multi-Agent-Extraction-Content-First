"""
layer_1/agent_1b_state.py — Agent 1B Internal State

Simple state definition for the Unstructured Text Parser.
"""

from typing import Any, Dict, List, Optional, TypedDict


class Agent1BState(TypedDict, total=False):
    # Input
    file_path: str

    # Output
    markdown_chunks: Optional[List[Dict[str, Any]]]
    """Each chunk dict:
        {
            "chunk_id" : int,
            "content"  : str,               # clean text (prose or pipe-delimited table)
            "metadata" : {
                "headings"     : list[str],  # e.g., ["2.1 ST01_FILLING"]
                "page_numbers" : list[int],  # (empty for now)
                "is_table"     : bool,
                "chunk_type"   : str,        # "table" or "prose"
            },
            "char_count": int,
        }
    """
    chunk_count: Optional[int]

    # Control flow
    status: str               # "pending" → "parsed" → "complete" or "error"
    error_message: Optional[str]