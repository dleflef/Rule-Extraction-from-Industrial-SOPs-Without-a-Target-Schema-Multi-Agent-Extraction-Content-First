"""
layer_1/protocol.py — Shared output protocol for Layer 1 agents.

All Layer 1 agents (1A, 1B, 1C) must normalise their final state into this
standard dict before handing results back to Layer 0. This guarantees that
evaluate_subordinate_output always receives a consistent structure regardless
of which agent ran.
"""


def normalize_layer1_output(
    agent_name: str,
    status: str,
    total_records: int,
    data_preview: str,
    error_message: str = None,
) -> dict:
    """
    Returns a standardised summary dict for a completed Layer 1 agent run.

    Args:
        agent_name    : Human-readable agent identifier (e.g. "Agent 1A").
        status        : Terminal status string (e.g. "complete", "error").
        total_records : Number of rows / chunks / log records produced.
        data_preview  : Brief human-readable description of the output content.
        error_message : Set when status == "error"; None on success.

    Returns:
        dict with keys: agent, status, total_records, data_preview, error_message.
    """
    return {
        "agent":         agent_name,
        "status":        status,
        "total_records": total_records,
        "data_preview":  data_preview,
        "error_message": error_message,
    }
