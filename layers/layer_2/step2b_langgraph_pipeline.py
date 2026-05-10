"""
step2b_langgraph_pipeline.py
================================
Agentic LangGraph Extractor/Judge pipeline for SOP rule extraction.

Architecture (cyclical critic pattern):

    ┌─────────────┐     always     ┌───────────┐
    │  extractor  │ ─────────────► │   judge   │
    └─────────────┘                └───────────┘
           ▲                             │
           │  errors found              │  no errors → END
           └─────────────────────────────┘  (or max iterations reached → END)

- Extractor node : LLM extracts rules; on retry it receives the judge's
                   feedback and fixes its mistakes.
- Judge node     : deterministic Python validation against the rule schema
                   (missing fields, invalid classes, bad numeric values,
                    sensor-naming pattern, duplicate IDs).
- Router         : loops back if errors remain and iteration < MAX_ITERATIONS,
                   otherwise finishes with the best available extraction.

Output CSVs share the same schema as step2 results and are evaluated
directly by step3_evaluate_extraction.py.

Usage (run from the layers/ directory)
-----
    python layer_2/step2b_langgraph_pipeline.py
    python layer_2/step2b_langgraph_pipeline.py --model qwen2.5:7b
    python layer_2/step2b_langgraph_pipeline.py --force

Output
------
    step2_results/ext_<model>_langgraph_run1.csv
    state/grid_search_metadata.csv
    state/experiment_registry.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "qwen2.5:7b"
PARADIGM_NAME   = "langgraph"

MAX_ITERATIONS  = 3       # max extractor/judge loops per SOP document
SOP_TEXT_LIMIT  = 8000    # max chars fed to the model per document

LLM_TIMEOUT_SEC = 600
LLM_NUM_CTX     = 8192
MAX_OUTPUT_TOKENS = 8192
MAX_RETRIES     = 3
RETRY_BASE_DELAY = 15.0

TEXTS_DIR     = "texts"
RESULTS_DIR   = "step2_results"
STATE_DIR     = "state"
REGISTRY_FILE = os.path.join(STATE_DIR, "experiment_registry.json")
METADATA_FILE = os.path.join(STATE_DIR, "grid_search_metadata.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR,   exist_ok=True)

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
client = OpenAI(api_key="ollama", base_url=_OLLAMA_BASE_URL, timeout=LLM_TIMEOUT_SEC)

NO_SYSTEM_ROLE: set[str] = {"mistralai/mistral-7b-instruct-v0.3", "mistral:7b"}


def _apply_no_system_role(model: str, messages: list[dict]) -> list[dict]:
    """Merge system message into first user message for models that don't support system role."""
    if model not in NO_SYSTEM_ROLE:
        return messages
    result = []
    system_content = ""
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            if system_content and m["role"] == "user":
                result.append({"role": "user", "content": f"{system_content}\n\n{m['content']}"})
                system_content = ""
            else:
                result.append(m)
    return result


# ── TOKEN TRACKING ────────────────────────────────────────────────────────────


class LLMUsage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens    = prompt
        self.completion_tokens = completion
        self.total_tokens     = prompt + completion

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            self.prompt_tokens    + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


_token_acc: LLMUsage = LLMUsage()


def _reset_tokens() -> None:
    global _token_acc
    _token_acc = LLMUsage()


def _add_tokens(u: LLMUsage) -> None:
    global _token_acc
    _token_acc = _token_acc + u


def _get_tokens() -> LLMUsage:
    return _token_acc


# ── LLM CALL WITH RETRY ───────────────────────────────────────────────────────


def llm_call(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> tuple[str, LLMUsage]:
    """Single LLM API call with exponential-backoff retry."""
    messages = _apply_no_system_role(model, messages)
    last_exc: Exception | None = None
    delay = RETRY_BASE_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            print(
                f"      [LLM] {model} attempt {attempt}/{MAX_RETRIES} "
                f"| ~{prompt_chars} chars",
                flush=True,
            )
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"options": {"num_ctx": LLM_NUM_CTX}},
            )
            content = resp.choices[0].message.content or ""
            usage = LLMUsage()
            if resp.usage is not None:
                usage = LLMUsage(
                    prompt=getattr(resp.usage, "prompt_tokens", 0) or 0,
                    completion=getattr(resp.usage, "completion_tokens", 0) or 0,
                )
            _add_tokens(usage)
            print(
                f"      [LLM] done in {time.time()-t0:.1f}s "
                f"| in={usage.prompt_tokens} out={usage.completion_tokens} tokens",
                flush=True,
            )
            return content, usage

        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                print(f"      [LLM] attempt {attempt} failed: {exc}. Retry in {delay:.0f}s…")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"      [LLM] all {MAX_RETRIES} attempts failed: {exc}")

    raise last_exc  # type: ignore[misc]


# ── JSON PARSING ──────────────────────────────────────────────────────────────


def parse_rules(raw: str) -> list[dict]:
    """
    Multi-strategy JSON extraction from a raw LLM reply.
    Mirrors the battle-tested parser in step2 (think-block stripping,
    truncation repair, regex fallback, object-level salvage).
    """
    if not raw:
        return []

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    def _extract(text: str) -> list[dict]:
        data = json.loads(text)
        rules = data.get("rules", data) if isinstance(data, dict) else data
        return [r for r in rules if isinstance(r, dict)]

    # Strategy 1: direct parse
    try:
        return _extract(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: repair truncated JSON (last '}' + close open brackets)
    last_close = raw.rfind("}")
    if last_close != -1:
        cand = raw[: last_close + 1]
        sq = cand.count("[") - cand.count("]")
        cu = cand.count("{") - cand.count("}")
        if sq >= 0 and cu >= 0:
            try:
                return _extract(cand + "]" * sq + "}" * cu)
            except (json.JSONDecodeError, KeyError):
                pass

    # Strategy 3: regex extraction
    for pat in [
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        r"(\{[^{}]*\"rules\".*\})",
        r"(\[.*\])",
    ]:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try:
                return _extract(m.group(1))
            except (json.JSONDecodeError, KeyError):
                continue

    # Strategy 4: salvage individual rule objects
    salvaged = []
    for m in re.finditer(r'\{[^{}]*"ruleId"[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                salvaged.append(obj)
        except json.JSONDecodeError:
            continue
    return salvaged


# ── PROMPTS ───────────────────────────────────────────────────────────────────

_BASE = (
    "You are a Precision Data Engineer specialised in SCADA systems and industrial SOPs. "
    "Your task is to extract technical rules from the provided text. "
    'Reply EXCLUSIVELY with a JSON: {"rules": [{"ruleId": "...", "class": "...", '
    '"station": "...", "sensor": "...", "sensorType": "...", "condition": "...", '
    '"action": "...", "severity": "...", "critHi": "...", "warnHi": "...", '
    '"warnLo": "...", "critLo": "...", "unit": "..."}, ...]}. '
    "Valid classes: ThresholdRule, OperationalRule, MaintenanceRule, AccessRule, AnomalyRule. "
    "Valid severities: CRITICAL, WARNING, MANDATORY, INFO, ADVISORY. "
    "For ThresholdRule extract ALL numeric values (critHi, warnHi, warnLo, critLo, unit). "
    'For all other classes leave numeric fields empty (""). '
    "Extract every single rule and table row — do not summarise or omit. "
    "MANDATORY: 'sensor' MUST follow pattern STATION_ID_SENSORTYPE "
    "(e.g. ST01_FILLING_TMP, WRH01_WAREHOUSE_HUM). "
    "'station' contains only the station ID (e.g. ST01_FILLING)."
)

EXTRACTION_SYSTEM = _BASE

REFINEMENT_SYSTEM = (
    _BASE
    + " You are in REFINEMENT MODE: a validation judge found errors in your previous "
    "extraction. Fix every listed error while keeping all correct rules unchanged."
)

# The {feedback} placeholder is filled at runtime
REFINEMENT_USER_TEMPLATE = (
    "A validation judge found the following errors in your previous extraction:\n\n"
    "{feedback}\n\n"
    "Fix ALL errors listed above. Keep every rule that was already correct. "
    "Reply exclusively with the corrected, complete JSON.\n\n"
    "Original SOP text:\n{sop_text}\n\n"
    "Your previous extraction (for reference — do not copy blindly):\n{prev_json}"
)


# ── JUDGE VALIDATION LOGIC ───────────────────────────────────────────────────

VALID_CLASSES     = {"ThresholdRule", "OperationalRule", "MaintenanceRule", "AccessRule", "AnomalyRule"}
VALID_SEVERITIES  = {"CRITICAL", "WARNING", "MANDATORY", "INFO", "ADVISORY"}

# e.g. ST01_FILLING_TMP, WRH01_WAREHOUSE_HUM, SRV01_SERVERROOM_CPU
_SENSOR_RE = re.compile(r"^[A-Z]{2,4}\d{2}_[A-Z]+_[A-Z]{2,5}$")


def validate_rules(rules: list[dict]) -> list[str]:
    """
    Deterministic schema validation.
    Returns a list of human-readable error strings; empty list = all rules pass.

    Checks performed:
      1. At least one rule was extracted
      2. No duplicate ruleIds
      3. class is one of VALID_CLASSES
      4. condition, action, station are non-empty
      5. severity is valid (if provided)
      6. sensor follows STATION_ID_SENSORTYPE pattern (if provided)
      7. ThresholdRule: critHi, critLo, warnHi, warnLo are present and numeric; unit present
    """
    if not rules:
        return ["No rules extracted — output was empty or could not be parsed as JSON."]

    errors: list[str] = []
    seen_ids: set[str] = set()

    for r in rules:
        rid = str(r.get("ruleId", "") or "UNKNOWN").strip()
        cls = str(r.get("class",  "") or "").strip()

        # 1. Duplicate ruleId
        nid = rid.lower()
        if nid in seen_ids:
            errors.append(f"Duplicate ruleId: '{rid}'")
        seen_ids.add(nid)

        # 2. Valid class
        if cls not in VALID_CLASSES:
            errors.append(
                f"{rid}: class='{cls}' is invalid. "
                f"Must be one of: {', '.join(sorted(VALID_CLASSES))}"
            )

        # 3. Required text fields
        for field in ("condition", "action", "station"):
            if not str(r.get(field, "") or "").strip():
                errors.append(f"{rid}: '{field}' is empty")

        # 4. Severity (only checked if provided)
        sev = str(r.get("severity", "") or "").strip().upper()
        if sev and sev not in VALID_SEVERITIES:
            errors.append(
                f"{rid}: severity='{sev}' is invalid. "
                f"Use one of: {', '.join(sorted(VALID_SEVERITIES))}"
            )

        # 5. Sensor naming pattern (only checked if provided)
        sensor = str(r.get("sensor", "") or "").strip()
        if sensor and not _SENSOR_RE.match(sensor):
            errors.append(
                f"{rid}: sensor='{sensor}' does not match the required pattern "
                "STATION_ID_SENSORTYPE (e.g. ST01_FILLING_TMP, WRH01_WAREHOUSE_HUM)"
            )

        # 6. ThresholdRule numeric fields
        if cls == "ThresholdRule":
            missing, bad = [], []
            for num_f in ("critHi", "critLo", "warnHi", "warnLo"):
                val = str(r.get(num_f, "") or "").strip()
                if not val:
                    missing.append(num_f)
                else:
                    try:
                        float(re.sub(r"[^\d.\-]", "", val))
                    except ValueError:
                        bad.append(f"{num_f}='{val}'")
            if missing:
                errors.append(
                    f"{rid}: ThresholdRule missing numeric fields: {', '.join(missing)}"
                )
            if bad:
                errors.append(
                    f"{rid}: ThresholdRule non-numeric value(s): {', '.join(bad)}"
                )
            if not str(r.get("unit", "") or "").strip():
                errors.append(f"{rid}: ThresholdRule missing 'unit' field")

    return errors


# ── GRAPH STATE ───────────────────────────────────────────────────────────────


class GraphState(TypedDict):
    """State that travels through the LangGraph pipeline for one SOP document."""
    raw_text:        str          # SOP excerpt (≤ SOP_TEXT_LIMIT chars)
    sop_filename:    str          # e.g. "SOP_001_OperatingProcedures.txt"
    model:           str          # Ollama model tag
    extracted_rules: List[Dict]   # current best extraction
    critic_feedback: str          # validation errors formatted as a bullet list
    iteration:       int          # number of completed extractor calls
    llm_turns:       int          # total LLM API calls made


# ── NODES ─────────────────────────────────────────────────────────────────────


def extractor_node(state: GraphState) -> dict:
    """
    Calls the LLM to extract (or re-extract) rules.

    Iteration 0 → clean extraction with the base system prompt.
    Iteration N → refinement: judge feedback is embedded in the user message
                  alongside the original text and the previous extraction.
    """
    text      = state["raw_text"]
    model     = state["model"]
    feedback  = state.get("critic_feedback", "")
    iteration = state.get("iteration", 0)
    turns     = state.get("llm_turns", 0)

    if iteration == 0:
        print(
            f"\n  [Extractor] Iteration 1 — initial extraction "
            f"from '{state['sop_filename']}'",
            flush=True,
        )
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user",   "content": f"SOP text:\n{text}"},
        ]
    else:
        prev_rules = state.get("extracted_rules", [])
        prev_json  = json.dumps({"rules": prev_rules}, ensure_ascii=False, indent=2)
        n_errors   = len([l for l in feedback.splitlines() if l.strip()])
        print(
            f"\n  [Extractor] Iteration {iteration + 1} — refinement "
            f"({n_errors} error(s) to fix)",
            flush=True,
        )
        messages = [
            {"role": "system", "content": REFINEMENT_SYSTEM},
            {"role": "user",   "content": REFINEMENT_USER_TEMPLATE.format(
                feedback=feedback,
                sop_text=text,
                prev_json=prev_json,
            )},
        ]

    try:
        raw, _ = llm_call(model, messages)
        rules  = parse_rules(raw)
        print(f"  [Extractor] Parsed {len(rules)} rules", flush=True)
    except Exception as exc:
        print(f"  [Extractor] LLM call failed: {exc} — keeping previous extraction")
        rules = state.get("extracted_rules", [])

    return {
        "extracted_rules": rules,
        "iteration":       iteration + 1,
        "llm_turns":       turns + 1,
        # critic_feedback is NOT reset here; the judge will overwrite it
    }


def judge_node(state: GraphState) -> dict:
    """
    Validates the current extraction against the schema.
    Produces a structured error list that the extractor can act on.
    Returns empty critic_feedback when all rules pass.
    """
    rules     = state.get("extracted_rules", [])
    iteration = state.get("iteration", 0)

    errors = validate_rules(rules)

    if errors:
        feedback = "\n".join(f"- {e}" for e in errors)
        print(
            f"  [Judge] {len(errors)} validation error(s) after iteration {iteration}:",
            flush=True,
        )
        for e in errors[:6]:
            print(f"    • {e}", flush=True)
        if len(errors) > 6:
            print(f"    … and {len(errors) - 6} more", flush=True)
    else:
        feedback = ""
        print(
            f"  [Judge] {len(rules)} rules — all passed validation ✓",
            flush=True,
        )

    return {"critic_feedback": feedback}


# ── CONDITIONAL EDGE ──────────────────────────────────────────────────────────


def should_continue(state: GraphState) -> str:
    """
    Routing logic after the judge:
      - No errors          → end (clean output)
      - Errors + room left → continue (loop back to extractor)
      - Errors + max hit   → end (best-effort output)
    """
    feedback  = state.get("critic_feedback", "")
    iteration = state.get("iteration", 0)

    if not feedback:
        print(f"  [Router] Clean output — done in {iteration} iteration(s)", flush=True)
        return "end"

    if iteration >= MAX_ITERATIONS:
        print(
            f"  [Router] Max iterations ({MAX_ITERATIONS}) reached — "
            "saving best-effort output",
            flush=True,
        )
        return "end"

    print(
        f"  [Router] Errors remain — looping back "
        f"(iteration {iteration}/{MAX_ITERATIONS})",
        flush=True,
    )
    return "continue"


# ── GRAPH CONSTRUCTION & COMPILATION ─────────────────────────────────────────

_workflow = StateGraph(GraphState)
_workflow.add_node("extractor", extractor_node)
_workflow.add_node("judge",     judge_node)

_workflow.set_entry_point("extractor")
_workflow.add_edge("extractor", "judge")
_workflow.add_conditional_edges(
    "judge",
    should_continue,
    {"continue": "extractor", "end": END},
)

pipeline = _workflow.compile()


# ── RESULT SAVING ─────────────────────────────────────────────────────────────

RULE_FIELDS = [
    "ruleId", "class", "station", "sensor", "sensorType",
    "condition", "action", "severity",
    "critHi", "warnHi", "warnLo", "critLo", "unit",
    "source_file", "model_name", "paradigm", "level", "run_id",
    "llm_turns", "text_truncated", "langgraph_iterations",
]


def save_results(run_id: str, rules: list[dict], out_dir: str) -> str:
    """Save extracted rules to a CSV compatible with step3 evaluation."""
    if not rules:
        print(f"  No rules to save for {run_id}")
        return ""

    extra      = sorted({k for r in rules for k in r.keys()} - set(RULE_FIELDS))
    fieldnames = RULE_FIELDS + extra

    for r in rules:
        for num_f in ("critHi", "warnHi", "warnLo", "critLo"):
            val = re.sub(r"[°%a-zA-Z/\s]", "", str(r.get(num_f, "") or "").strip())
            r[num_f] = val
        for f in RULE_FIELDS:
            r.setdefault(f, "")

    out = os.path.join(out_dir, f"ext_{run_id}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rules)

    print(f"  {len(rules)} rules → {out}")
    return out


# ── METADATA LOGGING ──────────────────────────────────────────────────────────

_METADATA_FIELDS = [
    "run_id", "model", "paradigm", "level",
    "total_rules", "sop_files", "total_iterations",
    "duration_sec", "prompt_tokens", "completion_tokens", "total_tokens",
]


def append_metadata(record: dict) -> None:
    write_header = not os.path.exists(METADATA_FILE)
    with open(METADATA_FILE, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METADATA_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(record)


# ── REGISTRY ──────────────────────────────────────────────────────────────────


def _load_registry() -> dict[str, str]:
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(registry: dict[str, str]) -> None:
    tmp = REGISTRY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, REGISTRY_FILE)


# ── RUNNER ────────────────────────────────────────────────────────────────────


def _make_run_id(model: str) -> str:
    safe = model.replace(":", "-").replace(".", "_").replace("/", "-")
    return f"{safe}_{PARADIGM_NAME}_run1"


def run_pipeline(
    model:       str  = DEFAULT_MODEL,
    texts_dir:   str  = TEXTS_DIR,
    results_dir: str  = RESULTS_DIR,
    force:       bool = False,
) -> None:
    """
    Run the LangGraph pipeline over all SOP .txt files in texts_dir.
    Produces one ext_<run_id>.csv compatible with step3_evaluate_extraction.py.
    """
    os.makedirs(results_dir, exist_ok=True)

    abs_texts = os.path.abspath(texts_dir)
    if not os.path.isdir(abs_texts):
        print(f"SOP directory not found: {abs_texts}")
        return
    txt_files = sorted(f for f in os.listdir(abs_texts) if f.endswith(".txt"))
    if not txt_files:
        print(f"No .txt files found in: {abs_texts}")
        return

    run_id   = _make_run_id(model)
    registry = _load_registry()

    if not force and registry.get(run_id) == "completed":
        print(f"SKIP {run_id} — already completed (use --force to re-run)")
        return

    print(f"\n{'─'*65}")
    print(f"  LangGraph Extractor/Judge Pipeline")
    print(f"  model={model} | max_iterations={MAX_ITERATIONS} | run_id={run_id}")
    print(f"  SOP files: {txt_files}")
    print(f"{'─'*65}")

    _reset_tokens()
    t_start          = time.time()
    all_rules:  list[dict] = []
    total_iterations = 0

    for filename in txt_files:
        filepath = os.path.join(abs_texts, filename)
        print(f"\n── {filename} ──────────────────────────────────────────────")
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                text = fh.read()

            truncated = len(text) > SOP_TEXT_LIMIT
            excerpt   = text[:SOP_TEXT_LIMIT]

            initial_state: GraphState = {
                "raw_text":        excerpt,
                "sop_filename":    filename,
                "model":           model,
                "extracted_rules": [],
                "critic_feedback": "",
                "iteration":       0,
                "llm_turns":       0,
            }

            final_state      = pipeline.invoke(initial_state)
            rules            = final_state["extracted_rules"]
            iterations_used  = final_state["iteration"]
            turns_used       = final_state["llm_turns"]
            total_iterations += iterations_used

            print(
                f"\n  → {len(rules)} rules | {iterations_used} iteration(s) "
                f"| {turns_used} LLM call(s)",
                flush=True,
            )

            for r in rules:
                r.update(
                    source_file=filename,
                    run_id=run_id,
                    model_name=model,
                    paradigm=PARADIGM_NAME,
                    level=2,                 # classified as L2 (multi-call loop)
                    text_truncated=truncated,
                    llm_turns=turns_used,
                    langgraph_iterations=iterations_used,
                )
            all_rules.extend(rules)

        except Exception as exc:
            print(f"  ERROR processing {filename}: {exc}")

    duration = time.time() - t_start
    tokens   = _get_tokens()

    save_results(run_id, all_rules, results_dir)
    append_metadata({
        "run_id":            run_id,
        "model":             model,
        "paradigm":          PARADIGM_NAME,
        "level":             2,
        "total_rules":       len(all_rules),
        "sop_files":         len(txt_files),
        "total_iterations":  total_iterations,
        "duration_sec":      round(duration, 2),
        "prompt_tokens":     tokens.prompt_tokens,
        "completion_tokens": tokens.completion_tokens,
        "total_tokens":      tokens.total_tokens,
    })

    registry[run_id] = "completed"
    _save_registry(registry)

    print(f"\n{'='*65}")
    print(f"Pipeline complete: {len(all_rules)} total rules | {duration:.0f}s")
    print(f"  Tokens in={tokens.prompt_tokens} out={tokens.completion_tokens}")
    print(f"  Results: {results_dir}/ext_{run_id}.csv")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangGraph Extractor/Judge agentic SOP rule extractor"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--texts-dir",
        default=TEXTS_DIR,
        help=f"Directory containing SOP .txt files (default: {TEXTS_DIR})",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"Output directory for CSV results (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if already marked completed in the registry",
    )
    args = parser.parse_args()

    run_pipeline(
        model=args.model,
        texts_dir=args.texts_dir,
        results_dir=args.results_dir,
        force=args.force,
    )
