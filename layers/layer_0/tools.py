import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from .utils import safe_preview_file

# Module-level LLM for the validation tool — instantiated once so that
# get_openai_callback() in main.py can track its invocations correctly.
_validation_llm = ChatOpenAI(
    model="meta-llama-3.1-8b-instruct",
    temperature=0.0,
    api_key="lm-studio-local",
    base_url="http://127.0.0.1:1234/v1",
)

# Layer 1 imports — loaded once at module initialisation so the LangGraph
# compilation cost is paid only on the first import, not per tool call.
from layer_1.agent_1a import agent_1a_app
from layer_1.agent_1b import agent_1b_app
from layer_1.agent_1c import agent_1c_app
from layer_1.protocol import normalize_layer1_output

# Stores the real result summary from route_to_layer_1 keyed by source_id so
# evaluate_subordinate_output can pass the actual output to the validation LLM.
_layer1_results: dict = {}

@tool
def inspect_source(file_path: str) -> str:
    """Use this to safely read the first few lines of a file to understand its schema or content."""
    return safe_preview_file(file_path)

@tool
def route_to_layer_1(source_id: str, file_path: str, agent_type: str, priority: int) -> str:
    """
    Routes a source to a Layer 1 ingestion agent and physically invokes it.

    Args:
        source_id  : The inventory identifier for this source (e.g. 'SRC_001').
        file_path  : The file_path value taken directly from the inventory entry.
        agent_type : 'structured_parser'   → invokes Agent 1A (CSV / JSON).
                     'unstructured_parser' → PDF / text (mocked; Layer 1B TBD).
                     'signal_parser'       → time-series signals (mocked; Layer 1C TBD).
        priority   : 1 for structured sources, 2 for unstructured sources.
    """
    # ── Structured files: invoke Agent 1A ────────────────────────────────────
    if agent_type == "structured_parser":
        try:
            # Build the full initial state for the Agent 1A graph.
            initial_state = {
                "file_path":        file_path,
                "raw_schema":       None,
                "standardized_data": None,
                "status":           "pending",
                "error_message":    None,
            }

            final_state = agent_1a_app.invoke(initial_state)

            # Compose a rich result string for the Layer 0 evaluation tool.
            status = final_state.get("status", "unknown")

            if status in {"complete", "standardized"}:
                std = final_state.get("standardized_data", {}) or {}
                total_rows = std.get("total_rows", "N/A")
                file_type  = (final_state.get("raw_schema") or {}).get("file_type", "unknown")
                result_str = (
                    f"SUCCESS: {source_id} processed by Agent 1A (structured_parser). "
                    f"Priority {priority}. "
                    f"Rows processed: {total_rows}. "
                    f"File type: {file_type}."
                )
                _layer1_results[source_id] = json.dumps(normalize_layer1_output(
                    agent_name="Agent 1A",
                    status=status,
                    total_records=total_rows if isinstance(total_rows, int) else 0,
                    data_preview=f"file_type={file_type}, rows={total_rows}",
                ))
                return result_str
            else:
                # Agent ran but ended in an error state — still report details
                err = final_state.get("error_message", "unknown error")
                return (
                    f"PARTIAL: {source_id} routed to Agent 1A (structured_parser) "
                    f"but ingestion ended with status='{status}'. "
                    f"Error: {err}"
                )

        except Exception as exc:
            return (
                f"ERROR: Agent 1A invocation failed for {source_id}. "
                f"Reason: {exc}"
            )

    # ── Unstructured files (PDF, DOCX, PPTX, HTML): invoke Agent 1B ─────────
    elif agent_type == "unstructured_parser":
        try:
            initial_state = {
                "file_path":       file_path,
                "raw_markdown":    None,
                "page_count":      None,
                "markdown_chunks": None,
                "chunk_count":     None,
                "status":          "pending",
                "error_message":   None,
            }

            final_state = agent_1b_app.invoke(initial_state)
            status = final_state.get("status", "unknown")

            if status == "complete":
                chunks      = final_state.get("markdown_chunks", [])
                chunk_count = final_state.get("chunk_count", len(chunks))
                page_count  = final_state.get("page_count", "N/A")
                first_meta  = chunks[0].get("metadata", {}) if chunks else {}
                top_section = first_meta.get("h1", first_meta.get("h2", "N/A"))
                result_str = (
                    f"SUCCESS: {source_id} processed by Agent 1B (unstructured_parser). "
                    f"Priority {priority}. "
                    f"Pages parsed: {page_count}. "
                    f"Semantic chunks produced: {chunk_count}. "
                    f"First section: '{top_section}'."
                )
                _layer1_results[source_id] = json.dumps(normalize_layer1_output(
                    agent_name="Agent 1B",
                    status=status,
                    total_records=chunk_count if isinstance(chunk_count, int) else 0,
                    data_preview=f"pages={page_count}, chunks={chunk_count}, first_section='{top_section}'",
                ))
                return result_str
            else:
                err = final_state.get("error_message", "unknown error")
                return (
                    f"PARTIAL: {source_id} routed to Agent 1B (unstructured_parser) "
                    f"but ingestion ended with status='{status}'. "
                    f"Error: {err}"
                )

        except Exception as exc:
            return (
                f"ERROR: Agent 1B invocation failed for {source_id}. "
                f"Reason: {exc}"
            )

    # ── Signal / log files: invoke Agent 1C ──────────────────────────────────
    elif agent_type == "signal_parser":
        try:
            initial_state = {
                "file_path":        file_path,
                "sample_lines":     None,
                "structural_regex": None,
                "standardized_logs": None,
                "status":           "pending",
                "error_message":    None,
            }

            final_state = agent_1c_app.invoke(initial_state)
            status = final_state.get("status", "unknown")

            if status in {"complete", "standardized"}:
                logs          = final_state.get("standardized_logs", {}) or {}
                parsed_count  = logs.get("parsed_records_count", 0)
                total_lines   = logs.get("total_lines_processed", "N/A")
                result_str = (
                    f"SUCCESS: {source_id} processed by Agent 1C (signal_parser). "
                    f"Priority {priority}. "
                    f"Parsed records: {parsed_count}. "
                    f"Total lines processed: {total_lines}."
                )
                _layer1_results[source_id] = json.dumps(normalize_layer1_output(
                    agent_name="Agent 1C",
                    status=status,
                    total_records=parsed_count if isinstance(parsed_count, int) else 0,
                    data_preview=f"parsed_records={parsed_count}, total_lines={total_lines}",
                ))
                return result_str
            else:
                err = final_state.get("error_message", "unknown error")
                return (
                    f"PARTIAL: {source_id} routed to Agent 1C (signal_parser) "
                    f"but ingestion ended with status='{status}'. "
                    f"Error: {err}"
                )

        except Exception as exc:
            return (
                f"ERROR: Agent 1C invocation failed for {source_id}. "
                f"Reason: {exc}"
            )

    else:
        return (
            f"ERROR: Unknown agent_type '{agent_type}' for {source_id}. "
            f"Valid types: structured_parser, unstructured_parser, signal_parser."
        )

@tool
def evaluate_subordinate_output(source_id: str) -> str:
    """
    Layer 0 supervisory QA: checks ingestion-readiness of a Layer 1 agent's output.
    Evaluates only parse quality and downstream completeness for Layer 2 — not ontology
    or KG semantics (those are deferred to Layer 2 / Layer 3).
    Call this after routing a source to verify that ingestion completed correctly.
    """
    layer_1_output = _layer1_results.get(
        source_id, "No ingestion report available for this source."
    )

    evaluation_prompt = f"""
    You are the Layer 0 Ingestion Quality Evaluator in a Knowledge Graph construction pipeline.
    Review the following ingestion report produced by a Layer 1 Ingestion Agent for source {source_id}.
    Layer 1 is responsible ONLY for parsing raw files and producing structured output — it does NOT
    perform entity extraction, ontology alignment, or KG validation.

    Your task is to assess whether this Layer 1 output is ready for downstream processing (Layer 2).
    Do NOT evaluate ontology, semantic coherence, or KG consistency — those are handled later.

    Ingestion report to evaluate:
    {layer_1_output}

    Evaluate across four ingestion-quality dimensions:
    - parse_success_confidence: Was the raw file parsed completely without data loss or corruption? (0.0-1.0)
    - output_completeness: Are all expected output fields present and non-empty? (0.0-1.0)
    - format_validity: Is the output format consistent and well-structured for downstream use? (0.0-1.0)
    - downstream_readiness: Is this output ready to be passed to Layer 2 without pre-processing? (0.0-1.0)

    Then compute overall_quality_score as the simple average of the four dimensions.

    Return a valid JSON object EXACTLY matching this schema — no extra fields, no markdown:
    {{
      "parse_success_confidence": <float 0.0-1.0>,
      "output_completeness": <float 0.0-1.0>,
      "format_validity": <float 0.0-1.0>,
      "downstream_readiness": <float 0.0-1.0>,
      "overall_quality_score": <float 0.0-1.0>,
      "status": "success" or "failed",
      "reason": "brief explanation covering parse quality and downstream readiness"
    }}

    Set status to "failed" if overall_quality_score < 0.5, otherwise "success".
    """

    try:
        response = _validation_llm.invoke(evaluation_prompt)
        raw = response.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        # Fallback: compute overall_quality_score ourselves if the LLM omitted it
        if not isinstance(parsed.get("overall_quality_score"), (int, float)):
            scores = [
                parsed.get("parse_success_confidence", 0.0),
                parsed.get("output_completeness", 0.0),
                parsed.get("format_validity", 0.0),
                parsed.get("downstream_readiness", 0.0),
            ]
            parsed["overall_quality_score"] = round(sum(scores) / len(scores), 4)
        # Ensure status aligns with overall_quality_score if the LLM omitted it
        if "status" not in parsed:
            parsed["status"] = "success" if parsed["overall_quality_score"] >= 0.5 else "failed"
        return json.dumps(parsed)
    except Exception as e:
        return json.dumps({
            "parse_success_confidence": 0.0,
            "output_completeness": 0.0,
            "format_validity": 0.0,
            "downstream_readiness": 0.0,
            "overall_quality_score": 0.0,
            "status": "failed",
            "reason": f"Layer 0 QA evaluator stalled. Error: {str(e)}"
        })

@tool
def escalate_issue(source_id: str, reason: str) -> str:
    """Escalates a source that repeatedly fails evaluation."""
    return f"ESCALATED: {source_id}. Reason logged: {reason}."

# Map tools for the LangGraph node execution
tools = [inspect_source, route_to_layer_1, evaluate_subordinate_output, escalate_issue]
tools_by_name = {t.name: t for t in tools}