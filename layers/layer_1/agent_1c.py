"""
layer_1/agent_1c.py — Agent 1C: Sensor & Signal Mapper (LangGraph)

Implements the Layer 1 Ingestion pipeline for raw log and signal files.
Uses a hybrid approach: an LLM generates a structural regex from a tiny 
sample (O(1) tokens), and Python applies it in bulk (O(N) compute).

This agent strictly standardizes data; semantic ontology mapping is 
deferred to Layer 2.
"""

import json
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .agent_1c_state import Agent1CState
from .agent_1c_tools import extract_log_sample, standardize_logs_with_regex

# ─────────────────────────────────────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    model="Qwen2.5-7B-Instruct-GGUF",
    temperature=0.0,
    api_key="lm-studio-local",
    base_url="http://127.0.0.1:1234/v1",
)

# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Log Sampler (Programmatic, Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def sampler_node(state: Agent1CState) -> dict:
    """Stage 1: Grabs a small sample of the log file for the LLM to inspect."""
    print(f"[Agent 1C | Stage 1] Sampling log file: {state['file_path']}")
    result = extract_log_sample(state["file_path"])
    
    if "error" in result:
        print(f"[Agent 1C | Stage 1] ERROR: {result['error']}")
        return {"status": "error", "error_message": result["error"]}
        
    print(f"[Agent 1C | Stage 1] Successfully extracted {len(result['sample_lines'])} sample lines.")
    return {"sample_lines": result["sample_lines"], "status": "sampled"}

# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Structural Regex Generator (LLM Call, Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def regex_generator_node(state: Agent1CState) -> dict:
    """
    Stage 2: Prompts the LLM to write a regex with generic structural 
    named capture groups (e.g., timestamp, level, module, message).
    Crucially, it does NOT attempt to map these to the domain ontology.
    """
    if state.get("status") == "error":
        return {"status": "error"}

    sample_text = "\n".join(state["sample_lines"])
    
    prompt = f"""You are an expert data engineer tasked with creating a structural parser for raw log files.

=== LOG SAMPLE ===
{sample_text}

=== YOUR TASK ===
Write a single Python Regular Expression (regex) that successfully matches the structure of these log lines. 
You MUST use named capture groups `(?P<name>pattern)` to extract the logical columns of the log.

Rules:
1. Use generic, structural names for your capture groups (e.g., `timestamp`, `log_level`, `module`, `process_id`, `message`).
2. DO NOT use domain-specific or ontology terms. Just describe the literal structure of the line.

Return a VALID JSON OBJECT ONLY. No markdown fences or explanations.
Format example:
{{"regex_pattern": "^\\\\[(?P<timestamp>.*?)\\\\] \\\\[(?P<log_level>.*?)\\\\] \\\\[(?P<module>.*?)\\\\] (?P<message>.*)$"}}

Your JSON:"""

    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        raw_text = response.content.strip()
        
        # Robustly extract JSON
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")
            
        json_str = raw_text[start:end]
        data = json.loads(json_str)
        
        regex_pattern = data.get("regex_pattern")
        if not regex_pattern:
            raise ValueError("JSON missing 'regex_pattern' key.")
            
        print(f"[Agent 1C | Stage 2] Generated Structural Regex: {regex_pattern}")
        return {"structural_regex": regex_pattern, "status": "regex_generated"}

    except Exception as exc:
        err = f"LLM Regex generation failed: {exc}"
        print(f"[Agent 1C | Stage 2] ERROR: {err}")
        return {"status": "error", "error_message": err}

# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Regex Validator (Programmatic, Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

def regex_validator_node(state: Agent1CState) -> dict:
    """
    Stage 3: Tests the generated regex against the sample lines.
    Requires at least 80% of sample lines to match; otherwise aborts.
    """
    if state.get("status") == "error":
        return {"status": "error"}

    pattern = state["structural_regex"]
    sample_lines = state["sample_lines"]

    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        err = f"Generated regex is invalid: {exc}"
        print(f"[Agent 1C | Stage 3] ERROR: {err}")
        return {"status": "error", "error_message": err}

    matched = sum(1 for line in sample_lines if compiled.search(line))
    total = len(sample_lines)
    match_rate = matched / total if total > 0 else 0.0

    if match_rate < 0.8:
        err = (
            f"Regex matched only {matched}/{total} sample lines "
            f"({match_rate:.0%}), below the 80% threshold."
        )
        print(f"[Agent 1C | Stage 3] ERROR: {err}")
        return {"status": "error", "error_message": err}

    print(f"[Agent 1C | Stage 3] Regex validated — {matched}/{total} lines matched ({match_rate:.0%}).")
    return {"status": "regex_validated"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Bulk Standardizer (Programmatic, Stage 4)
# ─────────────────────────────────────────────────────────────────────────────

def standardization_node(state: Agent1CState) -> dict:
    """
    Stage 4: Applies the structural regex to the entire log file to create
    a normalized list of dictionaries.
    """
    print(f"[Agent 1C | Stage 4] Executing bulk log standardization.")

    result = standardize_logs_with_regex(state["file_path"], state["structural_regex"])

    if "error" in result:
        print(f"[Agent 1C | Stage 4] ERROR: {result['error']}")
        return {"status": "error", "error_message": result["error"]}

    print(
        f"[Agent 1C | Stage 4] Standardization complete — "
        f"Parsed {result['parsed_records_count']} records out of "
        f"{result['total_lines_processed']} total lines. "
        f"Failed to parse: {result['unparsed_count']} lines."
    )
    
    return {"standardized_logs": result, "status": "complete"}

# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge & Graph Compilation
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_sampler(state: Agent1CState) -> str:
    return "end" if state.get("status") == "error" else "regex_generator"

def _route_after_regex_generator(state: Agent1CState) -> str:
    return "end" if state.get("status") == "error" else "regex_validator"

def _route_after_validator(state: Agent1CState) -> str:
    return "end" if state.get("status") == "error" else "standardize"

_workflow = StateGraph(Agent1CState)

_workflow.add_node("sampler", sampler_node)
_workflow.add_node("regex_generator", regex_generator_node)
_workflow.add_node("regex_validator", regex_validator_node)
_workflow.add_node("standardize", standardization_node)

_workflow.set_entry_point("sampler")

_workflow.add_conditional_edges(
    "sampler",
    _route_after_sampler,
    {"regex_generator": "regex_generator", "end": END},
)
_workflow.add_conditional_edges(
    "regex_generator",
    _route_after_regex_generator,
    {"regex_validator": "regex_validator", "end": END},
)
_workflow.add_conditional_edges(
    "regex_validator",
    _route_after_validator,
    {"standardize": "standardize", "end": END},
)
_workflow.add_edge("standardize", END)

agent_1c_app = _workflow.compile()