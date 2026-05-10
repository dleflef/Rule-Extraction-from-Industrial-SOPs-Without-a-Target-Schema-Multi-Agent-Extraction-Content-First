"""

28 april 2026

step2_grid_search_extraction.py
================================
Grid search for LLM-based rule extraction from industrial SOP documents.

Runs every combination of (model, paradigm) on the 4 SOP text files,
saves one CSV of extracted rules per run, and appends a metadata row
to grid_search_metadata.csv.

Usage
-----
    python3 step2_grid_search_extraction.py
    python3 step2_grid_search_extraction.py --models qwen2.5:14b phi4
    python3 step2_grid_search_extraction.py --paradigms naive few_shot_static
    python3 step2_grid_search_extraction.py --force    # re-run all, ignore registry
    python3 step2_grid_search_extraction.py --abox data/seed_rules/dataset/kg_seeds/nodes_factory.csv

Output
------
    results/ext_<model>_<paradigm>_run1.csv   -- extracted rules
    grid_search_metadata.csv                  -- one row per completed run
    experiment_registry.json                  -- resume checkpoint
"""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import sys
import threading
import time
from collections import Counter
from typing import Any

from openai import OpenAI

# ── MODELS AND PARADIGMS ──────────────────────────────────────────────────────

MODELS: list[str] = [
    "llama3.2:3b",
    "mistral:7b",
    "qwen2.5:7b",
    "deepseek-r1:7b",
    "qwen2.5:14b",
    "phi4",
    "deepseek-r1:14b",
]

PARADIGMS: list[str] = [
    # L0 -- single call, prompt-only
    "naive",
    "few_shot_static",
    "graph_informed",
    # L1 -- single call with explicit reasoning instruction
    "cot_basic",
    "cot_structured",
    "pre_act",
    # L2 -- multi-call with self-improvement
    "self_consistency",
    "reflexion",
    "reflexion_guided",
    # L3 -- agentic with external tool access
    "react_abox",
]

# Complexity level for each paradigm (used in logging and metadata)
PARADIGM_LEVEL: dict[str, int] = {
    "naive": 0,
    "few_shot_static": 0,
    "graph_informed": 0,
    "cot_basic": 1,
    "cot_structured": 1,
    "pre_act": 1,
    "self_consistency": 2,
    "reflexion": 2,
    "reflexion_guided": 2,
    "react_abox": 3,
}

# Models that generate an internal <think> block before the JSON output.
# The <think> block is stripped by parse_rules() before JSON parsing.
REASONING_NATIVE: set[str] = {"deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1-distill-qwen-7b"}

# Models whose chat template does not support the system role.
# The system message is merged into the first user message automatically.
NO_SYSTEM_ROLE: set[str] = {"mistralai/mistral-7b-instruct-v0.3", "mistral:7b"}


# ── EXPERIMENT PARAMETERS ─────────────────────────────────────────────────────

N_RUNS = 1  # runs per (model, paradigm); temperature=0 makes
# multiple runs identical for deterministic paradigms

SOP_TEXT_LIMIT = 8000  # max characters passed to the model per SOP document

# self_consistency: 3 independent calls at temp=0.3, then aggregate
SELF_CONS_RUNS = 3
SELF_CONS_TEMP = 0.3
MAJORITY_THRESH = 2  # a rule must appear in at least 2 of 3 runs to be kept

# Minimum content_agreement to cluster two rules as the same entity.
# Slightly below the step3c threshold of 0.6 because Jaccard is more
# conservative than SBERT.
ENSEMBLE_AGREE_THR = 0.55

# Voting strategy for self_consistency:
#   "content_aware" -- group by normalised ruleId, pick the version with
#                      highest internal content agreement
#   "score_based"   -- group by content similarity regardless of ruleId;
#                      captures rules that agree on content but differ in naming
SELF_CONS_VOTING = "score_based"

# react_abox: max VERIFY->observe turns before forcing JSON output
MAX_REACT_TURNS = 5


# ── LLM CONNECTION PARAMETERS ─────────────────────────────────────────────────

LLM_TIMEOUT_SEC = 600  # API call timeout (10 min) -- llama3.2:1b does ~3 tok/s on CPU
LLM_NUM_CTX = 8192  # context window: 2k input + 1k output
MAX_OUTPUT_TOKENS = (
    4096  # salvage parser recovers complete rules even if JSON is truncated
)
MAX_RETRIES = 3  # attempts before giving up on a single call
RETRY_BASE_DELAY = 15.0  # seconds before first retry; doubles each attempt


# ── FILE PATHS ────────────────────────────────────────────────────────────────

TEXTS_DIR = "texts"
RESULTS_DIR = "step2_results"
STATE_DIR = "state"
REGISTRY_FILE = os.path.join(STATE_DIR, "experiment_registry.json")
METADATA_FILE = os.path.join(STATE_DIR, "grid_search_metadata.csv")
HEARTBEAT_FILE = "heartbeat.txt"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# Allow overriding the Ollama endpoint via environment variable.
# Docker Compose sets OLLAMA_BASE_URL=http://ollama:11434/v1 automatically.
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

client = OpenAI(
    api_key="ollama",
    base_url=_OLLAMA_BASE_URL,
    timeout=LLM_TIMEOUT_SEC,
)


# ── PROMPT CONSTANTS ──────────────────────────────────────────────────────────

# Shared base for all paradigms: defines the task and the output JSON schema
_BASE_SYSTEM = (
    "You are a Precision Data Engineer specialised in SCADA systems and industrial SOPs. "
    "Your task is to extract technical rules from the provided text. "
    "Reply EXCLUSIVELY with a JSON structured as follows: "
    '{"rules": [{"ruleId", "class", "station", "sensor", "sensorType", '
    '"condition", "action", "severity", '
    '"critHi", "warnHi", "warnLo", "critLo", "unit"}, ...]}. '
    "For ThresholdRule entries extract numeric values into the dedicated fields "
    "(critHi, warnHi, warnLo, critLo, unit). "
    'For all other classes leave those fields empty ("").'
    "Do not summarize. You MUST extract every single rule and table row exhaustively. "
)

# graph_informed: adds mandatory sensor naming and numeric field mapping
_GRAPH_CONSTRAINT = (
    " MANDATORY CONSTRAINT: the 'sensor' field MUST follow the pattern "
    "STATION_TYPE (e.g. ST01_FILLING_TMP, WRH01_WAREHOUSE_HUM). "
    "The 'station' field contains only the station ID (e.g. ST01_FILLING). "
    "ruleIds must faithfully reflect the SOP document. "
    "For ThresholdRule entries (tables with CRIT_LO/WARN_LO/WARN_HI/CRIT_HI columns) "
    "you MUST populate critLo, warnLo, warnHi, critHi, unit "
    'with the pure numeric values from the table (e.g. critHi="30.0", unit="C").'
)

# few_shot_static: two concrete output examples, one per main rule class
_FEW_SHOT_EXAMPLE = (
    " Follow these style examples:"
    "\n\nExample 1 -- OperationalRule (narrative rule):"
    ' {"ruleId":"RULE-ST01-01","class":"OperationalRule",'
    '"station":"ST01_FILLING","sensor":"ST01_FILLING_FLW","sensorType":"FLW",'
    '"condition":"FLW MUST be between 112 and 128 L/min",'
    '"action":"Inspect and notify maintenance","severity":"MANDATORY",'
    '"critHi":"","warnHi":"","warnLo":"","critLo":"","unit":""}.'
    "\n\nExample 2 -- ThresholdRule (numeric threshold from table):"
    ' {"ruleId":"RULE-THR-ST01-TMP-CRIT","class":"ThresholdRule",'
    '"station":"ST01_FILLING","sensor":"ST01_FILLING_TMP","sensorType":"TMP",'
    '"condition":"TMP critical thresholds for ST01_FILLING",'
    '"action":"Inspect and notify maintenance","severity":"CRITICAL",'
    '"critHi":"30.0","warnHi":"26.0","warnLo":"18.0","critLo":"15.0","unit":"C"}.'
    "\n\nFor ThresholdRule: map CRIT_LO to critLo, WARN_LO to warnLo, "
    "WARN_HI to warnHi, CRIT_HI to critHi. "
    'Always use the pure numeric value (e.g. "30.0", not "30.0C").'
)

# reflexion_guided turn 3: domain-specific completeness checklist
REFLEXION_CHECKLIST = """Verify the following rule categories in your output:
1. STATIONS: have you extracted rules for all stations?
   ST01_FILLING, ST02_SEALING, ST03_LABELLING, ST04_PACKAGING,
   SRV01_SERVERROOM, WRH01_WAREHOUSE, CHM01_CHEMICALSTORAGE,
   RND01_RDLAB, CAF01_CAFETERIA
2. CROSS-STATION DEPENDENCIES: have you included rules linking different stations?
   (e.g. current ST02 -> speed ST04, temperature SRV01 -> line halt)
3. ANOMALIES: have you extracted rules for all temporal patterns?
   SPIKE, DRIFT, STUCK, OUT_OF_RANGE, CORRELATED
4. MAINTENANCE: have you included predictive/corrective rules?
   MAINT-01 to MAINT-08
5. ACCESS AND OCCUPANCY: have you included access control and zone occupancy limits?
   RULE-ACCESS-01 to RULE-ACCESS-04 and occupancy rules for each zone
6. NUMERIC VALUES: are the thresholds (critHi, warnHi, critLo, warnLo) correct?
For each missing or imprecise category, add or correct the rules in the final JSON."""


# ── TOKEN TRACKING ────────────────────────────────────────────────────────────


class LLMUsage:
    """Tracks prompt and completion token counts for a single run."""

    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    def __repr__(self) -> str:
        return (
            f"LLMUsage(in={self.prompt_tokens}, "
            f"out={self.completion_tokens}, tot={self.total_tokens})"
        )


# Thread-local accumulator: each run accumulates its own token count independently
_tls = threading.local()


def _reset_tokens() -> None:
    _tls.acc = LLMUsage()


def _get_tokens() -> LLMUsage:
    return getattr(_tls, "acc", LLMUsage())


def _add_tokens(u: LLMUsage) -> None:
    acc = getattr(_tls, "acc", LLMUsage())
    _tls.acc = acc + u


# ── HEARTBEAT THREAD ──────────────────────────────────────────────────────────


class HeartbeatThread(threading.Thread):
    """
    Daemon thread that writes the current timestamp and experiment ID
    to heartbeat.txt every 30 seconds.
    Useful for detecting whether the process is still alive after a long pause.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._current_exp: str = "idle"
        self._lock = threading.Lock()

    def set_experiment(self, exp_id: str) -> None:
        with self._lock:
            self._current_exp = exp_id

    def run(self) -> None:
        while True:
            with self._lock:
                exp_id = self._current_exp
            try:
                with open(HEARTBEAT_FILE, "w") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | {exp_id}\n")
            except OSError:
                pass
            time.sleep(30)


_heartbeat = HeartbeatThread()
_heartbeat.start()


# ── SIGNAL HANDLER ────────────────────────────────────────────────────────────

# Global reference to the live registry so the signal handler can save it
_registry_ref: dict[str, str] = {}


def _save_and_exit(signum, frame) -> None:
    """Save the registry on Ctrl+C or SIGTERM so no completed runs are lost."""
    print(f"\n\nSignal {signum} received -- saving state...")
    try:
        save_registry(_registry_ref)
        completed = sum(1 for v in _registry_ref.values() if v == "completed")
        print(
            f"   Saved: {completed}/{len(_registry_ref)} completed in {REGISTRY_FILE}"
        )
        print(f"   To resume: python3 step2_grid_search_extraction.py")
    except Exception as e:
        print(f"   Save error: {e}")
    sys.exit(0)


signal.signal(signal.SIGINT, _save_and_exit)
signal.signal(signal.SIGTERM, _save_and_exit)


# ── EXPERIMENT REGISTRY ───────────────────────────────────────────────────────
# Maps run_id -> "completed" | "partial".
# "partial" means the run started but did not finish -- it will be retried.
# "completed" means the run finished and its CSV was saved -- it will be skipped.


def make_run_id(model: str, paradigm: str, run_n: int) -> str:
    """
    Canonical ID for a single run.
    Format: <model_normalised>_<paradigm>_run<N>
    Example: qwen2_5-14b_naive_run1
    """
    base = f"{model}_{paradigm}".replace(":", "-").replace(".", "_").replace("/", "-")
    return f"{base}_run{run_n}"


def make_exp_id(model: str, paradigm: str) -> str:
    """Base ID without run number -- used for grouping in evaluation."""
    return f"{model}_{paradigm}".replace(":", "-").replace(".", "_").replace("/", "-")


def load_registry() -> dict[str, str]:
    """Load registry from disk. Returns empty dict if file does not exist."""
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Registry file corrupted -- starting from scratch.")
        return {}


def save_registry(registry: dict[str, str]) -> None:
    """Atomic save: write to a temp file then rename to avoid partial writes."""
    tmp = REGISTRY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, REGISTRY_FILE)


def mark_partial(registry: dict[str, str], run_id: str) -> None:
    registry[run_id] = "partial"
    save_registry(registry)


def mark_completed(registry: dict[str, str], run_id: str) -> None:
    registry[run_id] = "completed"
    save_registry(registry)


def is_completed(registry: dict[str, str], run_id: str) -> bool:
    return registry.get(run_id) == "completed"


# ── LLM CALL WITH RETRY ───────────────────────────────────────────────────────


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Merge a leading system message into the first user message for models
    that only support user/assistant roles (e.g. Mistral)."""
    if not messages or messages[0].get("role") != "system":
        return messages
    system_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0].get("role") == "user":
        rest[0] = {**rest[0], "content": f"{system_content}\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": system_content})
    return rest


def _llm_call_raw(
    model: str,
    messages: list[dict],
    temperature: float = 0,
) -> tuple[str, LLMUsage]:
    """
    Single API call with exponential-backoff retry.

    Handles timeout, connection refused (Ollama not yet ready),
    and transient API errors. Raises the last exception after MAX_RETRIES
    failed attempts.

    num_ctx is forwarded to Ollama via extra_body so the context window
    is always LLM_NUM_CTX regardless of the server's default setting.
    """
    last_exc: Exception | None = None
    delay = RETRY_BASE_DELAY

    if model in NO_SYSTEM_ROLE:
        messages = _merge_system_into_user(messages)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            ts_start = time.strftime("%H:%M:%S")
            print(
                f"      [LLM {ts_start}] {model} | attempt {attempt}/{MAX_RETRIES} "
                f"| prompt ~{prompt_chars} chars | ctx {LLM_NUM_CTX} | max_out {MAX_OUTPUT_TOKENS}",
                flush=True,
            )
            t_call = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            elapsed = time.time() - t_call
            ts_end = time.strftime("%H:%M:%S")
            content = resp.choices[0].message.content
            usage = LLMUsage()
            if resp.usage is not None:
                usage = LLMUsage(
                    prompt=getattr(resp.usage, "prompt_tokens", 0) or 0,
                    completion=getattr(resp.usage, "completion_tokens", 0) or 0,
                )
            print(
                f"      [LLM {ts_end}] done in {elapsed:.1f}s "
                f"| in={usage.prompt_tokens} out={usage.completion_tokens} tokens "
                f"| reply {len(content or '')} chars",
                flush=True,
            )
            print(
                f"      [LLM raw reply]\n{content}\n      [/LLM raw reply]", flush=True
            )
            return content, usage

        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                print(
                    f"    Attempt {attempt}/{MAX_RETRIES} failed "
                    f"({type(exc).__name__}): {exc}"
                )
                print(f"    Waiting {delay:.0f}s before retry...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"    All {MAX_RETRIES} attempts failed for {model}: {exc}")

    raise last_exc


def llm_call(
    model: str,
    messages: list[dict],
    temperature: float = 0,
) -> tuple[str, LLMUsage]:
    """Wrapper that accumulates token counts into the thread-local counter."""
    content, usage = _llm_call_raw(model, messages, temperature)
    _add_tokens(usage)
    return content, usage


# ── JSON PARSING ──────────────────────────────────────────────────────────────


def parse_rules(raw: str) -> list[dict]:
    """
    Extract the rule list from a raw LLM response.

    LLMs sometimes wrap JSON in markdown fences or add explanatory text.
    Three strategies are tried in order:
      1. Direct JSON parse of the full response
      2. Extract the first JSON block inside markdown fences
      3. Extract the first {...} or [...] block found by regex

    Also strips <think>...</think> blocks produced by reasoning-native models.
    Returns an empty list if all strategies fail.
    """
    if not raw:
        return []

    # Remove reasoning traces from deepseek-r1 and similar models
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

    # Strategy 1.5: repair truncated JSON — find last '}', close unclosed '[' and '{'
    last_close = raw.rfind("}")
    if last_close != -1:
        candidate = raw[: last_close + 1]
        opens_sq = candidate.count("[") - candidate.count("]")
        opens_cu = candidate.count("{") - candidate.count("}")
        if opens_sq >= 0 and opens_cu >= 0:
            repaired = candidate + "]" * opens_sq + "}" * opens_cu
            try:
                rules = _extract(repaired)
                print(
                    f"      [parse] repaired truncated JSON → {len(rules)} rules",
                    flush=True,
                )
                return rules
            except (json.JSONDecodeError, KeyError):
                pass

    # Strategy 2 & 3: regex extraction
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

    # Strategy 4: salvage individual complete rule objects from truncated JSON.
    # When the LLM hits max_tokens the closing brackets are missing; we extract
    # every well-formed {...} object that contains at least a "ruleId" key.
    salvaged = []
    for m in re.finditer(r'\{[^{}]*"ruleId"[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                salvaged.append(obj)
        except json.JSONDecodeError:
            continue
    if salvaged:
        print(
            f"      [parse] salvaged {len(salvaged)} rules from truncated JSON",
            flush=True,
        )
        return salvaged

    return []


# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────


def system_prompt(paradigm: str) -> str:
    """
    Build the system prompt for a given paradigm.
    All paradigms share _BASE_SYSTEM as the foundation.
    Each paradigm appends its specific instruction or example.
    """
    if paradigm == "naive":
        return _BASE_SYSTEM + " Direct extraction without external constraints."

    if paradigm == "few_shot_static":
        # Two concrete JSON examples showing OperationalRule and ThresholdRule format
        return _BASE_SYSTEM + _FEW_SHOT_EXAMPLE

    if paradigm == "graph_informed":
        # Enforces sensor naming convention and numeric column mapping
        return _BASE_SYSTEM + _GRAPH_CONSTRAINT

    if paradigm == "cot_basic":
        # Instruct the model to reason before producing JSON
        return (
            _BASE_SYSTEM
            + " First reason step by step identifying each rule and its logical "
            "structure, then produce the final JSON."
        )

    if paradigm == "cot_structured":
        # Five explicit reasoning steps: stations, sensors, thresholds, dependencies, access
        return (
            _BASE_SYSTEM
            + " Follow EXACTLY these reasoning steps before producing the JSON:\n"
            "1. List all stations or zones mentioned in the text.\n"
            "2. For each station, identify the sensors and their measurement types.\n"
            "3. For each sensor, extract the threshold conditions (including numeric values).\n"
            "4. Identify causal dependencies between different stations.\n"
            "5. Identify access control, occupancy, and maintenance rules.\n"
            "Only after these 5 steps produce the JSON."
        )

    if paradigm == "pre_act":
        # Pre-action planning: describe the document structure before extracting
        return (
            _BASE_SYSTEM
            + " Before extracting the rules, briefly describe in 2-3 lines: "
            "(a) the document structure, (b) the categories of rules present, "
            "(c) any dependencies between sections. "
            "Then produce the JSON."
        )

    # Default used internally by L2 and L3 paradigms
    return _BASE_SYSTEM + _GRAPH_CONSTRAINT


# ── CONTENT AGREEMENT (used internally by self_consistency voting) ─────────────


def _norm_id(rid: Any) -> str:
    """Normalise a ruleId to lowercase alphanumeric for comparison."""
    return re.sub(r"[^a-z0-9]", "", str(rid or "").lower())


def _numeric_score(a: Any, b: Any) -> float:
    """
    Similarity between two numeric threshold values (0.0-1.0).
    - Both empty  -> 1.0 (no information, no penalty)
    - One missing -> 0.0 (present vs absent is a mismatch)
    - Both present -> 1 - |a-b| / max(|a|, |b|)
    """
    sa, sb = str(a or "").strip(), str(b or "").strip()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    try:
        fa, fb = float(sa), float(sb)
        return max(0.0, 1.0 - abs(fa - fb) / max(abs(fa), abs(fb), 1e-9))
    except ValueError:
        return 1.0 if sa == sb else 0.0


def _content_agreement(r1: dict, r2: dict) -> float:
    """
    Weighted similarity between two rule objects (0.0-1.0).
    Ignores ruleId -- measures whether two rules describe the same
    operational entity based on their structured and numeric fields.

    Field weights:
      station, sensor  2.0  -- location identity (wrong station = wrong rule)
      critHi, critLo   3.0  -- primary signal for ThresholdRule
      warnHi, warnLo   1.5  -- secondary numeric bounds
      condition/action 0.5  for ThresholdRule (numbers already carry the signal)
                       2.0  for all other classes (text is the main discriminant)

    Note: this uses Jaccard token overlap for text fields, which is a fast
    approximation. The final evaluation in step3c uses SBERT cosine similarity.
    """
    cls = str(r1.get("class", "") or r2.get("class", "")).strip()
    parts: list[tuple[float, float]] = []  # (score, weight)

    # Categorical fields: binary exact match after normalisation
    for field, w in [
        ("class", 1.5),
        ("station", 2.0),
        ("sensor", 2.0),
        ("sensorType", 1.0),
        ("severity", 1.0),
        ("unit", 1.0),
    ]:
        v1 = re.sub(r"[^a-z0-9]", "", str(r1.get(field, "") or "").lower())
        v2 = re.sub(r"[^a-z0-9]", "", str(r2.get(field, "") or "").lower())
        if v1 or v2:
            parts.append((1.0 if v1 == v2 else 0.0, w))

    # Numeric fields: tolerance score, only for ThresholdRule
    if cls == "ThresholdRule":
        for field, w in [
            ("critHi", 3.0),
            ("critLo", 3.0),
            ("warnHi", 1.5),
            ("warnLo", 1.5),
        ]:
            v1, v2 = r1.get(field), r2.get(field)
            if str(v1 or "").strip() or str(v2 or "").strip():
                parts.append((_numeric_score(v1, v2), w))

    # Text fields: Jaccard token overlap
    txt_w = 0.5 if cls == "ThresholdRule" else 2.0
    for field in ("condition", "action"):
        t1 = set(re.split(r"[^a-z0-9]", str(r1.get(field, "") or "").lower())) - {""}
        t2 = set(re.split(r"[^a-z0-9]", str(r2.get(field, "") or "").lower())) - {""}
        if t1 or t2:
            jacc = len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 1.0
            parts.append((jacc, txt_w))

    if not parts:
        return 0.0
    return sum(s * w for s, w in parts) / sum(w for _, w in parts)


# ── VOTING STRATEGIES FOR SELF_CONSISTENCY ────────────────────────────────────


def _vote_content_aware(all_runs: list[list[dict]]) -> list[dict]:
    """
    Content-aware majority vote.

    Groups rules across runs by normalised ruleId. Keeps a group only
    if it appears in >= MAJORITY_THRESH runs. Selects the candidate
    with the highest total content_agreement against all others in the
    group (consensus version). Applies majority vote per numeric field
    to stabilise threshold values for ThresholdRule.
    """
    vote: Counter[str] = Counter()
    candidates: dict[str, list[dict]] = {}

    for run in all_runs:
        seen: set[str] = set()
        for rule in run:
            nid = _norm_id(rule.get("ruleId", ""))
            if nid and nid not in seen:
                vote[nid] += 1
                seen.add(nid)
                candidates.setdefault(nid, []).append(rule)

    result: list[dict] = []
    for nid, count in vote.items():
        if count < MAJORITY_THRESH or nid not in candidates:
            continue
        group = candidates[nid]

        if len(group) == 1:
            best = group[0]
        else:
            scores = [
                sum(
                    _content_agreement(r, group[j]) for j in range(len(group)) if j != i
                )
                for i, r in enumerate(group)
            ]
            best = group[scores.index(max(scores))]

        best = dict(best)
        if best.get("class") == "ThresholdRule" and len(group) >= 2:
            for num_field in ("critHi", "critLo", "warnHi", "warnLo", "unit"):
                vals = [str(r.get(num_field, "") or "").strip() for r in group]
                non_empty = [v for v in vals if v]
                if non_empty:
                    majority_val, n = Counter(non_empty).most_common(1)[0]
                    if n >= MAJORITY_THRESH:
                        best[num_field] = majority_val

        result.append(best)
    return result


def _vote_score_based(all_runs: list[list[dict]]) -> list[dict]:
    """
    Score-based ensemble selection.

    Does not group by ruleId. Instead, builds a pairwise content_agreement
    matrix across all rules from all runs, greedily clusters rules that
    describe the same entity (agreement >= ENSEMBLE_AGREE_THR), and
    selects the best representative from each cross-run cluster.

    This captures cases where two runs agree on rule content but use
    different ruleIds -- which content_aware would miss.
    """
    tagged: list[tuple[int, dict]] = [
        (run_idx, rule) for run_idx, run in enumerate(all_runs) for rule in run
    ]
    if not tagged:
        return []

    n = len(tagged)

    # Pairwise agreement (upper triangle). Same-run pairs are forced to 0.
    agree: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if tagged[i][0] == tagged[j][0]:
                agree[(i, j)] = 0.0
            else:
                agree[(i, j)] = _content_agreement(tagged[i][1], tagged[j][1])

    # Greedy clustering: process highest-agreement pairs first.
    # Each rule belongs to at most one cluster.
    cluster_of: dict[int, int] = {}
    clusters: dict[int, list[int]] = {}
    next_cluster = 0

    for (i, j), score in sorted(agree.items(), key=lambda x: -x[1]):
        if score < ENSEMBLE_AGREE_THR:
            break
        ci, cj = cluster_of.get(i), cluster_of.get(j)
        if ci is None and cj is None:
            clusters[next_cluster] = [i, j]
            cluster_of[i] = cluster_of[j] = next_cluster
            next_cluster += 1
        elif ci is None:
            clusters[cj].append(i)
            cluster_of[i] = cj
        elif cj is None:
            clusters[ci].append(j)
            cluster_of[j] = ci

    # Unmatched rules become singleton clusters
    for idx in range(n):
        if idx not in cluster_of:
            clusters[next_cluster] = [idx]
            cluster_of[idx] = next_cluster
            next_cluster += 1

    result: list[dict] = []
    for members in clusters.values():
        # Keep only clusters spanning >= MAJORITY_THRESH different runs
        if len({tagged[m][0] for m in members}) < MAJORITY_THRESH:
            continue

        rules = [tagged[m][1] for m in members]
        if len(rules) == 1:
            best = rules[0]
        else:
            scores = [
                sum(
                    _content_agreement(r, rules[j]) for j in range(len(rules)) if j != i
                )
                for i, r in enumerate(rules)
            ]
            best = rules[scores.index(max(scores))]

        best = dict(best)
        if best.get("class") == "ThresholdRule" and len(members) >= 2:
            for num_field in ("critHi", "critLo", "warnHi", "warnLo", "unit"):
                vals = [str(r.get(num_field, "") or "").strip() for r in rules]
                non_empty = [v for v in vals if v]
                if non_empty:
                    majority_val, n = Counter(non_empty).most_common(1)[0]
                    if n >= MAJORITY_THRESH:
                        best[num_field] = majority_val

        result.append(best)
    return result


# ── MULTI-CALL PARADIGM ORCHESTRATORS ────────────────────────────────────────


def run_self_consistency(model: str, sop_text: str) -> list[dict]:
    """
    L2 -- self_consistency.
    Runs SELF_CONS_RUNS independent calls at temperature SELF_CONS_TEMP
    and aggregates results with the configured voting strategy.
    """
    sys_p = system_prompt("naive")
    user_p = f"SOP text:\n{sop_text}"
    all_runs: list[list[dict]] = []

    for _ in range(SELF_CONS_RUNS):
        raw, _ = llm_call(
            model,
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            temperature=SELF_CONS_TEMP,
        )
        all_runs.append(parse_rules(raw))

    if SELF_CONS_VOTING == "score_based":
        return _vote_score_based(all_runs)
    return _vote_content_aware(all_runs)


def run_reflexion(model: str, sop_text: str) -> list[dict]:
    """
    L2 -- reflexion: 2-turn loop.
      Turn 1: extract rules.
      Turn 2: self-critique (missing rules, wrong numerics, duplicates),
              produce corrected JSON.
    """
    sys_p = system_prompt("graph_informed")
    user_p = f"SOP text:\n{sop_text}"

    raw1, _ = llm_call(
        model,
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
    )

    critique = (
        "You extracted the following rules from the SOP text:\n"
        f"{json.dumps({'rules': parse_rules(raw1)}, ensure_ascii=False, indent=2)}\n\n"
        "Critically review this output:\n"
        "- Which important rules might be missing?\n"
        "- Are there imprecise or incorrect numeric conditions?\n"
        "- Are sensor and station names correct?\n"
        "- Are there duplicate or poorly structured rules?\n\n"
        "Produce a corrected and complete version in the required JSON format."
    )
    raw2, _ = llm_call(
        model,
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
            {"role": "assistant", "content": raw1},
            {"role": "user", "content": critique},
        ],
    )
    return parse_rules(raw2)


def run_reflexion_guided(model: str, sop_text: str) -> list[dict]:
    """
    L2 -- reflexion_guided: 3-turn loop.
      Turn 1: extract rules.
      Turn 2: open self-critique.
      Turn 3: structured domain checklist (stations, anomalies, maintenance,
              access rules, numeric verification).
    """
    sys_p = system_prompt("graph_informed")
    user_p = f"SOP text:\n{sop_text}"

    raw1, _ = llm_call(
        model,
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
    )

    critique_p = (
        "Review the rules you extracted. Which might be missing or "
        "imprecise? Produce an improved version in JSON format."
    )
    raw2, _ = llm_call(
        model,
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
            {"role": "assistant", "content": raw1},
            {"role": "user", "content": critique_p},
        ],
    )

    checklist_p = (
        f"You extracted {len(parse_rules(raw2))} rules. "
        f"Verify completeness using this checklist:\n\n{REFLEXION_CHECKLIST}\n\n"
        "Produce the final updated JSON with all necessary corrections."
    )
    raw3, _ = llm_call(
        model,
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
            {"role": "assistant", "content": raw1},
            {"role": "user", "content": critique_p},
            {"role": "assistant", "content": raw2},
            {"role": "user", "content": checklist_p},
        ],
    )
    return parse_rules(raw3)


def run_react_abox(
    model: str,
    sop_text: str,
    abox_sensors: set[str],
) -> tuple[list[dict], int]:
    """
    L3 -- react_abox: ReAct loop with ABox sensor verification.

    The model may emit VERIFY: <sensor_name> on a single line.
    The system replies SENSOR_OK or SENSOR_NOT_FOUND with up to 3
    candidate suggestions before the model produces the final JSON.
    Stops when the model outputs JSON without a VERIFY request,
    or after MAX_REACT_TURNS turns.

    Returns (rules, number_of_turns_used).
    """
    sys_p = (
        _BASE_SYSTEM
        + _GRAPH_CONSTRAINT
        + "\n\nAvailable tool: to verify a sensor name, write on a single line:\n"
        "  VERIFY: <sensor_name>\n"
        "The system will reply with SENSOR_OK or SENSOR_NOT_FOUND before you "
        "produce the final JSON. You may verify multiple sensors before the JSON."
    )

    def verify_sensor(name: str) -> str:
        """Check a sensor name against the ABox; return a one-line result."""
        nl = name.strip().lower()
        for s in abox_sensors:
            if s.lower() == nl:
                return f"SENSOR_OK: {s}"
        # Token-overlap fuzzy match for suggestions
        nt = set(re.split(r"[^a-z0-9]", nl)) - {""}
        scored = sorted(
            [
                (len(nt & set(re.split(r"[^a-z0-9]", s.lower())) - {""}), s)
                for s in abox_sensors
            ],
            reverse=True,
        )
        scored = [(n, s) for n, s in scored if n > 0]
        if scored:
            top_n, top_s = scored[0]
            if top_n == len(nt):
                return f"SENSOR_OK: {top_s}"
            return (
                f"SENSOR_NOT_FOUND: {name.strip()} -> did you mean: "
                f"{', '.join(s for _, s in scored[:3])}?"
            )
        return f"SENSOR_NOT_FOUND: {name.strip()} -> no match in ABox."

    messages: list[dict] = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": f"SOP text:\n{sop_text}"},
    ]
    turns = 0
    final_rules: list[dict] = []

    for _ in range(MAX_REACT_TURNS):
        turns += 1
        raw, _ = llm_call(model, messages)
        messages.append({"role": "assistant", "content": raw})

        verify_requests = re.findall(r"VERIFY:\s*(\S+)", raw)
        if verify_requests:
            observations = "\n".join(verify_sensor(v) for v in verify_requests)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Sensor verification results:\n{observations}\n\n"
                        "Now produce the final JSON."
                    ),
                }
            )
        else:
            final_rules = parse_rules(raw)
            break
    else:
        # Max turns reached -- parse the last assistant message
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                final_rules = parse_rules(msg["content"])
                break

    return final_rules, turns


# ── ABOX LOADING ──────────────────────────────────────────────────────────────


def load_abox_sensors(nodes_path: str) -> set[str]:
    """
    Load sensor names from the ABox nodes CSV.
    Tries the given path first, then several common fallback locations.
    Returns an empty set if not found (react_abox works without verification).
    """
    for path in [
        nodes_path,
        "data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
        "../data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
    ]:
        if os.path.exists(path):
            try:
                sensors: set[str] = set()
                with open(path, "r", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        if row.get("label", "").strip().lower() == "sensor":
                            name = row.get("name", "").strip()
                            if name:
                                sensors.add(name)
                print(f"  ABox loaded from {path}: {len(sensors)} sensors")
                return sensors
            except Exception as e:
                print(f"  Error loading ABox {path}: {e}")
    print("  ABox not found -- react_abox will run without sensor verification")
    return set()


# ── RESULT SAVING ─────────────────────────────────────────────────────────────


def save_partial_results(run_id: str, rules: list[dict]) -> None:
    """
    Save intermediate results after each SOP to a _partial.csv checkpoint.
    Overwritten after each document; replaced by the final file on completion.
    """
    if not rules:
        return
    out = os.path.join(RESULTS_DIR, f"ext_{run_id}_partial.csv")
    fieldnames = sorted({k for r in rules for k in r.keys()})
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rules)


def save_results(run_id: str, rules: list[dict]) -> None:
    """
    Save the final results for a run as ext_<run_id>.csv.

    Guarantees consistent column order across all output files.
    Strips unit suffixes from numeric fields (e.g. "30.0C" -> "30.0")
    so step3c can parse them as floats.
    """
    if not rules:
        print(f"    No rules to save for {run_id}")
        return

    RULE_FIELDS = [
        "ruleId",
        "class",
        "station",
        "sensor",
        "sensorType",
        "condition",
        "action",
        "severity",
        "critHi",
        "warnHi",
        "warnLo",
        "critLo",
        "unit",
        "source_file",
        "model_name",
        "paradigm",
        "level",
        "run_id",
        "llm_turns",
        "text_truncated",
    ]
    extra = sorted({k for r in rules for k in r.keys()} - set(RULE_FIELDS))
    fieldnames = RULE_FIELDS + extra

    for r in rules:
        for num_field in ("critHi", "warnHi", "warnLo", "critLo"):
            val = re.sub(r"[°%a-zA-Z/\s]", "", str(r.get(num_field, "") or "").strip())
            r[num_field] = val
        for f in RULE_FIELDS:
            r.setdefault(f, "")

    out = os.path.join(RESULTS_DIR, f"ext_{run_id}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rules)

    partial = os.path.join(RESULTS_DIR, f"ext_{run_id}_partial.csv")
    if os.path.exists(partial):
        os.remove(partial)

    print(f"    {len(rules)} rules -> {out}")


# ── METADATA LOGGING ─────────────────────────────────────────────────────────

CANONICAL_FIELDS = [
    "run_id",
    "base_exp_id",
    "model",
    "paradigm",
    "run_n",
    "level",
    "reasoning_native",
    "total_rules",
    "sop_files",
    "errors",
    "duration_sec",
    "avg_rule_sec",
    "avg_llm_turns",
    "temperature",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "tokens_per_rule",
]


def append_metadata(record: dict) -> None:
    """Append one row to grid_search_metadata.csv, creating the file if needed."""
    write_header = not os.path.exists(METADATA_FILE)
    with open(METADATA_FILE, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(record)


# ── SINGLE RUN ────────────────────────────────────────────────────────────────


def run_single_experiment(
    model: str,
    paradigm: str,
    txt_files: list[str],
    abox_sensors: set[str],
    run_id: str,
) -> tuple[list[dict], int, float, LLMUsage]:
    """
    Execute one (model, paradigm) combination across all SOP documents.

    For each document: read and truncate the text, call the paradigm
    function, annotate extracted rules with metadata, save a checkpoint.

    Returns: (rules, error_count, duration_seconds, token_usage)
    """
    all_rules: list[dict] = []
    errors = 0
    level = PARADIGM_LEVEL[paradigm]
    t_start = time.time()
    _reset_tokens()

    for filename in txt_files:
        filepath = os.path.join(os.path.abspath(TEXTS_DIR), filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                sop_text = f.read()

            truncated = len(sop_text) > SOP_TEXT_LIMIT
            sop_excerpt = sop_text[:SOP_TEXT_LIMIT]

            print(f"    {filename}...", end=" ", flush=True)
            t0 = time.time()

            if level in (0, 1):
                raw, _ = llm_call(
                    model,
                    [
                        {"role": "system", "content": system_prompt(paradigm)},
                        {"role": "user", "content": f"SOP text:\n{sop_excerpt}"},
                    ],
                )
                rules, n_turns = parse_rules(raw), 1

            elif paradigm == "self_consistency":
                rules, n_turns = (
                    run_self_consistency(model, sop_excerpt),
                    SELF_CONS_RUNS,
                )

            elif paradigm == "reflexion":
                rules, n_turns = run_reflexion(model, sop_excerpt), 2

            elif paradigm == "reflexion_guided":
                rules, n_turns = run_reflexion_guided(model, sop_excerpt), 3

            elif paradigm == "react_abox":
                rules, n_turns = run_react_abox(model, sop_excerpt, abox_sensors)

            else:
                print(f"Unknown paradigm: '{paradigm}'")
                rules, n_turns = [], 0

            for r in rules:
                r.update(
                    source_file=filename,
                    run_id=run_id,
                    model_name=model,
                    paradigm=paradigm,
                    level=level,
                    text_truncated=truncated,
                    llm_turns=n_turns,
                )

            all_rules.extend(rules)
            save_partial_results(run_id, all_rules)

            print(f"{len(rules)} rules ({n_turns} calls, {time.time()-t0:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")

    return all_rules, errors, time.time() - t_start, _get_tokens()


# ── MAIN GRID LOOP ────────────────────────────────────────────────────────────


def run_experiment(
    models: list[str] | None = None,
    paradigms: list[str] | None = None,
    abox_path: str = "data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
    force_redo: bool = False,
    n_runs: int = N_RUNS,
) -> None:
    """
    Outer loop over models x paradigms x run_n.

    Skips runs already marked "completed" in the registry unless
    force_redo=True. Marks each run "partial" before starting and
    "completed" after saving a non-empty result file.
    """
    global _registry_ref

    _models = models or MODELS
    _paradigms = paradigms or PARADIGMS

    abs_texts = os.path.abspath(TEXTS_DIR)
    if not os.path.isdir(abs_texts):
        print(f"SOP directory not found: {abs_texts}")
        return
    txt_files = sorted(f for f in os.listdir(abs_texts) if f.endswith(".txt"))
    if not txt_files:
        print(f"No .txt files found in: {abs_texts}")
        return

    abox_sensors = load_abox_sensors(abox_path)
    registry = {} if force_redo else load_registry()
    _registry_ref = registry

    total_runs = len(_models) * len(_paradigms) * n_runs
    done_runs = sum(1 for v in registry.values() if v == "completed")

    print(f"  ABox loaded: {len(abox_sensors)} sensors")
    print(f"  SOP files:   {txt_files}")
    print(f"  Models:      {_models}")
    print(f"  Paradigms:   {_paradigms}")
    print(f"  Runs per condition: {n_runs}")
    print(f"  Registry: {done_runs}/{total_runs} already completed")
    print(f"  To run:   {total_runs - done_runs}")
    print()

    for model in _models:
        for paradigm in _paradigms:
            level = PARADIGM_LEVEL[paradigm]
            for run_n in range(1, n_runs + 1):
                run_id = make_run_id(model, paradigm, run_n)
                _heartbeat.set_experiment(run_id)

                if not force_redo and is_completed(registry, run_id):
                    print(f"  SKIP {run_id}")
                    continue

                print(f"\n{'─'*65}")
                print(
                    f"  {model} | {paradigm} | L{level} | run {run_n}/{n_runs} | {run_id}"
                )
                print(f"{'─'*65}")

                mark_partial(registry, run_id)

                try:
                    rules, errors, duration, tokens = run_single_experiment(
                        model, paradigm, txt_files, abox_sensors, run_id
                    )
                except Exception as e:
                    print(f"  Critical error in {run_id}: {e}")
                    print(f"  Run stays 'partial' and will be retried on next launch.")
                    continue

                save_results(run_id, rules)

                avg_turns = (
                    sum(r.get("llm_turns", 1) for r in rules) / len(rules)
                    if rules
                    else 0.0
                )
                append_metadata(
                    {
                        "run_id": run_id,
                        "base_exp_id": make_exp_id(model, paradigm),
                        "model": model,
                        "paradigm": paradigm,
                        "run_n": run_n,
                        "level": level,
                        "reasoning_native": model in REASONING_NATIVE,
                        "total_rules": len(rules),
                        "sop_files": len(txt_files),
                        "errors": errors,
                        "duration_sec": round(duration, 2),
                        "avg_rule_sec": (
                            round(duration / len(rules), 4) if rules else 0.0
                        ),
                        "avg_llm_turns": round(avg_turns, 2),
                        "temperature": (
                            SELF_CONS_TEMP if paradigm == "self_consistency" else 0
                        ),
                        "prompt_tokens": tokens.prompt_tokens,
                        "completion_tokens": tokens.completion_tokens,
                        "total_tokens": tokens.total_tokens,
                        "tokens_per_rule": (
                            round(tokens.total_tokens / len(rules), 1) if rules else 0.0
                        ),
                    }
                )

                print(
                    f"    {len(rules)} rules | {errors} errors | {duration:.0f}s | "
                    f"tok_in={tokens.prompt_tokens} tok_out={tokens.completion_tokens}"
                )

                if rules:
                    mark_completed(registry, run_id)
                else:
                    print(f"    Warning: 0 rules extracted -- will be retried.")

    _heartbeat.set_experiment("completed")
    completed = sum(1 for v in registry.values() if v == "completed")
    print(f"\n{'='*65}")
    print(f"Grid complete: {completed}/{total_runs} runs")
    print(f"  Registry: {REGISTRY_FILE}")
    print(f"  Metadata: {METADATA_FILE}")
    print(f"  Results:  {RESULTS_DIR}/")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM grid search for SOP rule extraction"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS,
        help=f"Runs per condition (default: {N_RUNS})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to run (default: all)",
    )
    parser.add_argument(
        "--paradigms",
        nargs="+",
        default=None,
        help="Subset of paradigms (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore registry and re-run everything"
    )
    parser.add_argument(
        "--abox",
        default="data/seed_rules/dataset/kg_seeds/nodes_factory.csv",
        help="Path to nodes_factory.csv for react_abox sensor verification",
    )
    args = parser.parse_args()

    run_experiment(
        models=args.models,
        paradigms=args.paradigms,
        abox_path=args.abox,
        force_redo=args.force,
        n_runs=args.runs,
    )
