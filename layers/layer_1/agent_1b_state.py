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
            "content"  : str,               # clean text
            "metadata" : {
                "headings"     : list[str],  # always empty — LLM interprets headings
                "page_numbers" : list[int],  # always empty — not extracted by Docling
            },
            "char_count": int,
        }
    """
    chunk_count: Optional[int]

    # Control flow
    status: str               # "pending" → "complete" or "error"
    error_message: Optional[str]