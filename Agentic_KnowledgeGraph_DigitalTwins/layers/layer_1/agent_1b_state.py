"""
layer_1/agent_1b_state.py — Agent 1B Internal State

Simple state definition for the Unstructured Text Parser.
"""

from typing import List, Optional, TypedDict


class Agent1BState(TypedDict, total=False):
    # ── Inputs ─────────────────────────────────────────────────────────────────
    file_path: str
    # Directory where the .txt file will be saved (step2 reads from here).
    # Defaults to "texts" (relative to the working directory of the caller).
    texts_dir: str

    # ── Outputs ────────────────────────────────────────────────────────────────
    markdown_chunks: Optional[List[str]]
    """List of plain‑text chunk strings.
       Tables are clean pipe Markdown, headings are kept with their content.
       The LLM interprets structure from the text itself; no metadata.
    """
    chunk_count: Optional[int]
    # Absolute path of the .txt file written to texts_dir.
    # step2_grid_search_extraction.py reads every .txt in its texts/ directory.
    output_txt_path: Optional[str]

    # ── Control flow ───────────────────────────────────────────────────────────
    status: str               # "pending" → "complete" or "error"
    error_message: Optional[str]