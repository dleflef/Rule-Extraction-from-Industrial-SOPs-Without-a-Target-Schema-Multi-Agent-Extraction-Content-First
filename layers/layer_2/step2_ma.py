"""
step2_ma.py
==================================================
Enhanced LangGraph multi-agent SOP extraction pipeline.

Combines the clean fan-in architecture (Annotated state + custom reducer)
from agentic_sop_extraction_pipeline.py with the domain-specific prompts
from step2_new1.py and reflexion checklist insights from the grid search.

Architecture (8 stages, with parallel extractors and smart LLM routing):
    DocumentTypeClassifier  (semantic LLM classification)
      ↓ conditional fan-out
    NarrativeRuleMiner  ┐
    TabularThresholdMiner├─── parallel extractors (fan-in via custom reducer)
    AccessMatrixMiner   ┘
    ConsensusMerger          (smart LLM merge only for 'mixed')
    KnowledgeRouter          (routes AuthorizationRule → graph_edges, others → merged_rules)
    Validator                (pre-judge FP/FN scrubber)
    QualityGapEvaluator      (judge loop, max MAX_JUDGE_RETRIES retries)
    IdentifierCanonicalizer

──────────────────────────────────────────────────
ABLATION STUDY
──────────────────────────────────────────────────
Four single-component ablation variants are supported via --ablation.
Each removes exactly one stage and replaces it with a passthrough or
direct edge where needed.  Output files are tagged so they never
overwrite the full-pipeline results.

  --ablation no_consensus_merger
      Removes : ConsensusMerger (LLM dedup/merge)
      Replaces: SimplePassthroughMerge (deterministic dedup, no LLM)
      Pipeline: Classifier → Miners → SimplePassthroughMerge
                → KnowledgeRouter → Validator → QualityGapEvaluator
                → IdentifierCanonicalizer
      Output  : abl_noCM_s<seed>_run<n>.csv
      Use when: measuring the contribution of LLM-based conflict
                resolution vs. rule-based deduplication alone.

  --ablation no_quality_gap_evaluator
      Removes : QualityGapEvaluator (judge loop + retries)
      Pipeline: Classifier → Miners → ConsensusMerger
                → KnowledgeRouter → Validator → IdentifierCanonicalizer
      Output  : abl_noQGE_s<seed>_run<n>.csv
      Use when: measuring how much the iterative judge loop improves
                recall (no retries, no gap feedback).

  --ablation no_identifier_canonicalizer
      Removes : IdentifierCanonicalizer (LLM ruleId normalization)
      Replaces: PassthroughNormalize (copies merged_rules → final_rules)
      Pipeline: Classifier → Miners → ConsensusMerger
                → KnowledgeRouter → Validator → QualityGapEvaluator
                → PassthroughNormalize
      Output  : abl_noIC_s<seed>_run<n>.csv
      Use when: measuring the impact of canonical ruleId assignment
                on downstream evaluation / F1 scoring.

  --ablation no_validator
      Removes : Validator (pre-judge FP/FN scrubber)
      Pipeline: Classifier → Miners → ConsensusMerger
                → KnowledgeRouter → QualityGapEvaluator
                → IdentifierCanonicalizer
      Output  : abl_noV_s<seed>_run<n>.csv
      Use when: measuring how many false positives / false negatives
                the validator catches before the judge loop runs.

Usage examples:
    # ── Full pipeline (baseline, no ablation) ──────────────────────
    python step2_ma.py                          # 1 run, seed 42
    python step2_ma.py --runs 3                 # 3 runs, seed 42
    python step2_ma.py --runs 3 --seeds 42 123  # 3 runs × 2 seeds
    python step2_ma.py --no-force               # skip already-completed runs

    # ── Ablation runs ──────────────────────────────────────────────
    python step2_ma.py --ablation no_consensus_merger
    python step2_ma.py --ablation no_quality_gap_evaluator
    python step2_ma.py --ablation no_identifier_canonicalizer
    python step2_ma.py --ablation no_validator
    python step2_ma.py --ablation no_validator --runs 3 --seeds 42 123

    # Run all four ablations back-to-back:
    for abl in no_consensus_merger no_quality_gap_evaluator \\
               no_identifier_canonicalizer no_validator; do
        python step2_ma.py --ablation $abl
    done
──────────────────────────────────────────────────
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
from typing import Annotated, Dict, List, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

# ── PATHS ──────────────────────────────────────────────────────────────────────

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

TEXTS_DIR      = os.path.join(_SCRIPT_DIR, "..", "texts")
RESULTS_DIR    = os.path.join(_SCRIPT_DIR, "agentic_results")
STATE_DIR      = os.path.join(_SCRIPT_DIR, "..", "state")
REGISTRY_FILE  = os.path.join(STATE_DIR, "agentic_registry.json")
METADATA_FILE  = os.path.join(STATE_DIR, "agentic_metadata.csv")
HEARTBEAT_FILE = os.path.join(_SCRIPT_DIR, "..", "heartbeat.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# ── MODEL ASSIGNMENTS (optimized from grid-search evaluation) ──────────────────

MODEL_CLASSIFIER   = os.environ.get("MODEL_CLASSIFIER",   "qwen/qwen3-4b-2507")
MODEL_NARRATIVE    = os.environ.get("MODEL_NARRATIVE",    "ministral-3:8b")      # few_shot: F1 0.7394
MODEL_TABULAR      = os.environ.get("MODEL_TABULAR",      "qwen/qwen3-4b-2507")  # graph_informed: F1 0.7305
MODEL_MATRIX       = os.environ.get("MODEL_MATRIX",       "ministral-3:14b")
MODEL_ORCHESTRATOR = os.environ.get("MODEL_ORCHESTRATOR", "ministral-3:14b")     # merger/judge/normalizer/refiner

# ── LLM PARAMETERS ────────────────────────────────────────────────────────────

LLM_TIMEOUT_SEC   = 600
LLM_NUM_CTX       = 16384
MAX_OUTPUT_TOKENS = 16384
MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 15.0
SEED              = 42
MAX_JUDGE_RETRIES = 2
SOP_TEXT_LIMIT    = 8000

# ── CONNECTIONS ────────────────────────────────────────────────────────────────

_OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL",   "http://localhost:11434/v1")
_OLLAMA_API_KEY    = os.environ.get("OLLAMA_API_KEY",    "ollama")
_LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
_LMSTUDIO_API_KEY  = os.environ.get("LMSTUDIO_API_KEY",  "lm-studio")
LMSTUDIO_MODELS    = set(m.strip() for m in os.environ.get("LMSTUDIO_MODELS", "").split(",") if m.strip())

# Models whose chat template does not support the system role
NO_SYSTEM_ROLE = {"ministral-3:3b", "ministral-3:8b", "ministral-3:14b"}

_ollama_client   = OpenAI(api_key=_OLLAMA_API_KEY,   base_url=_OLLAMA_BASE_URL,   timeout=LLM_TIMEOUT_SEC)
_lmstudio_client = OpenAI(api_key=_LMSTUDIO_API_KEY, base_url=_LMSTUDIO_BASE_URL, timeout=LLM_TIMEOUT_SEC)

def _client_for(model: str) -> OpenAI:
    return _lmstudio_client if model in LMSTUDIO_MODELS else _ollama_client

# ── TOKEN TRACKING ─────────────────────────────────────────────────────────────

class LLMUsage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    def __init__(self, prompt: int = 0, completion: int = 0):
        self.prompt_tokens     = prompt
        self.completion_tokens = completion
        self.total_tokens      = prompt + completion
    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(self.prompt_tokens + other.prompt_tokens,
                        self.completion_tokens + other.completion_tokens)
    def __repr__(self) -> str:
        return f"LLMUsage(in={self.prompt_tokens}, out={self.completion_tokens}, tot={self.total_tokens})"

_tls = threading.local()
def _reset_tokens() -> None:  _tls.acc = LLMUsage()
def _get_tokens()  -> LLMUsage: return getattr(_tls, "acc", LLMUsage())
def _add_tokens(u: LLMUsage) -> None: _tls.acc = getattr(_tls, "acc", LLMUsage()) + u

# ── HEARTBEAT ──────────────────────────────────────────────────────────────────

class HeartbeatThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._current_exp = "idle"
        self._lock = threading.Lock()
    def set_experiment(self, exp_id: str):
        with self._lock: self._current_exp = exp_id
    def run(self):
        while True:
            with self._lock: exp_id = self._current_exp
            try:
                with open(HEARTBEAT_FILE, "w") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | {exp_id}\n")
            except OSError:
                pass
            time.sleep(30)

_heartbeat = HeartbeatThread()
_heartbeat.start()

# ── SIGNAL HANDLER ─────────────────────────────────────────────────────────────

_registry_ref: dict[str, str] = {}

def _save_and_exit(signum, frame):
    print(f"\nSignal {signum} -- saving state...")
    try:
        _save_registry(_registry_ref)
        completed = sum(1 for v in _registry_ref.values() if v == "completed")
        print(f"Saved: {completed}/{len(_registry_ref)} completed runs")
    except Exception as e:
        print(f"Save error: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT,  _save_and_exit)
signal.signal(signal.SIGTERM, _save_and_exit)

# ── REGISTRY ───────────────────────────────────────────────────────────────────

def _load_registry() -> dict[str, str]:
    if not os.path.exists(REGISTRY_FILE): return {}
    try:
        with open(REGISTRY_FILE, "r") as f: return json.load(f)
    except Exception: return {}

def _save_registry(registry: dict[str, str]) -> None:
    tmp = REGISTRY_FILE + ".tmp"
    with open(tmp, "w") as f: json.dump(registry, f, indent=2)
    os.replace(tmp, REGISTRY_FILE)

def _mark_completed(registry: dict, run_id: str) -> None:
    registry[run_id] = "completed"; _save_registry(registry)

def _is_completed(registry: dict, run_id: str) -> bool:
    return registry.get(run_id) == "completed"

# ── LLM HELPERS ────────────────────────────────────────────────────────────────

def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Merge system message into first user message for models without system role."""
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0].get("role") == "user":
        rest[0] = {**rest[0], "content": f"{sys_content}\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": sys_content})
    return rest

def _llm_call_raw(model: str, messages: list[dict], temperature: float = 0.0) -> tuple[str, LLMUsage]:
    last_exc = None
    delay = RETRY_BASE_DELAY
    if model in NO_SYSTEM_ROLE:
        messages = _merge_system_into_user(messages)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            print(f"      [{model}] attempt {attempt}/{MAX_RETRIES} | ~{prompt_chars} chars", flush=True)
            t0 = time.time()
            resp = _client_for(model).chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"options": {"seed": SEED, "num_ctx": LLM_NUM_CTX}},
            )
            elapsed = time.time() - t0
            content = resp.choices[0].message.content or ""
            usage = LLMUsage()
            if resp.usage:
                usage = LLMUsage(
                    prompt=getattr(resp.usage, "prompt_tokens", 0) or 0,
                    completion=getattr(resp.usage, "completion_tokens", 0) or 0,
                )
            print(f"      [{model}] done in {elapsed:.1f}s | in={usage.prompt_tokens} out={usage.completion_tokens}", flush=True)
            return content, usage
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                print(f"    Attempt {attempt} failed ({type(exc).__name__}): {exc}. Retry in {delay:.0f}s...")
                time.sleep(delay); delay *= 2
            else:
                print(f"    All {MAX_RETRIES} attempts failed: {exc}")
    raise last_exc

def llm_call(model: str, messages: list[dict], temperature: float = 0.0) -> tuple[str, LLMUsage]:
    content, usage = _llm_call_raw(model, messages, temperature)
    _add_tokens(usage)
    return content, usage

# ── JSON PARSING ────────────────────────────────────────────────────────────────

def parse_rules(raw: str) -> list[dict]:
    if not raw: return []
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    def _extract(text: str) -> list[dict]:
        data = json.loads(text)
        rules = data.get("rules", data) if isinstance(data, dict) else data
        return [r for r in rules if isinstance(r, dict)]

    try: return _extract(raw.strip())
    except Exception: pass

    last_close = raw.rfind("}")
    if last_close != -1:
        candidate = raw[:last_close + 1]
        opens_sq = candidate.count("[") - candidate.count("]")
        opens_cu = candidate.count("{") - candidate.count("}")
        if opens_sq >= 0 and opens_cu >= 0:
            repaired = candidate + "]" * opens_sq + "}" * opens_cu
            try:
                rules = _extract(repaired)
                print(f"      [parse] repaired truncated JSON → {len(rules)} rules", flush=True)
                return rules
            except Exception: pass

    for pat in [r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", r"(\{[^{}]*\"rules\".*\})", r"(\[.*\])"]:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try: return _extract(m.group(1))
            except Exception: continue

    salvaged = []
    for m in re.finditer(r'\{[^{}]*"ruleId"[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict): salvaged.append(obj)
        except Exception: continue
    if salvaged:
        print(f"      [parse] salvaged {len(salvaged)} rules from truncated JSON", flush=True)
        return salvaged
    return []


# ── PROMPTS ────────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = (
    "Classify the industrial SOP document below into one of four types. "
    "Return ONLY valid JSON: {\"doc_type\": \"tabular\"} or {\"doc_type\": \"narrative\"} "
    "or {\"doc_type\": \"matrix\"} or {\"doc_type\": \"mixed\"}.\n\n"
    "Use the SEMANTIC meaning of the content — not surface formatting or column names:\n\n"
    "  tabular — the document's primary purpose is to define numeric operating limits "
    "for sensors or equipment: acceptable ranges, warning bands, critical thresholds. "
    "The dominant content is measurement-point rows with upper and lower bound values. "
    "Choose this even if short prose or anomaly definitions also appear.\n\n"
    "  matrix — the document's primary purpose is to govern personnel authorization, "
    "zone occupancy limits, or alarm acknowledgment time requirements. "
    "The dominant content is about who is allowed where, how many people, or how fast alarms must be acknowledged.\n\n"
    "  mixed — the document contains meaningful content belonging to more than one of the types above "
    "that must each be extracted separately to avoid data loss. "
    "For example: a predominantly narrative document that also contains a table of numeric sensor thresholds "
    "must be classified as mixed so both the prose rules AND the threshold values are extracted. "
    "Do not ignore secondary content — if two distinct types of rules must be extracted, classify as mixed.\n\n"
    "  narrative — everything else: procedural rules in prose, labeled operational constraints, "
    "maintenance schedules, or a document whose primary content is written instructions rather than tables."
)

NARRATIVE_MINER_SYSTEM = (
    "You are a rule extractor for industrial SOPs. "
    "Extract OperationalRule and MaintenanceRule instances from prose, labeled entries, and maintenance tables.\n\n"
    "WHAT COUNTS AS A RULE: any explicit operational constraint, procedural requirement, or maintenance "
    "trigger written in the document — whether labeled with a code, written as a prose sentence, "
    "or defined as a table row. Use your understanding of industrial operations to identify rules; "
    "do not rely on any particular label format or naming convention.\n\n"
    "RULE IDENTIFIER: if the document assigns an explicit identifier to a rule (a code, number, "
    "or label of any format), use it exactly as written. If no identifier is given, set ruleId to ''.\n\n"
    "GRANULARITY: each rule must represent a single actionable condition at one severity level "
    "with one required action. If a sentence or entry describes multiple distinct conditions "
    "(e.g. different severity tiers with different required actions), extract each as its own rule "
    "sharing the same ruleId. The downstream normalizer will disambiguate.\n\n"
    "FIELDS:\n"
    "  - station: the physical location or station the rule applies to, as named in the document.\n"
    "  - sensor: compose the identifier from the location name and the measurement type, "
    "joined with an underscore (e.g. LOCATION_PARAMTYPE). Leave empty if no specific sensor is referenced.\n"
    "  - sensorType: the measurement type (e.g. TMP, FLW, PRS). Leave empty if not applicable.\n"
    "  - condition: the triggering condition, stated as close to the source text as possible.\n"
    "  - action: the required response or corrective action.\n"
    "  - severity: MANDATORY, WARNING, HIGH, CRITICAL, or MEDIUM — infer from the document context.\n"
    "  - Leave critHi/warnHi/warnLo/critLo/unit empty for OperationalRule and MaintenanceRule.\n"
    "  - For cross-system fault or dependency entries: each entry is one MaintenanceRule "
    "whose condition describes the causal relationship between systems.\n"
    "  - Do NOT invent rules not grounded in the document text.\n\n"
    "SELF-CHECK before outputting: (a) have you covered every explicit rule or constraint in the document? "
    "(b) for each location, have you extracted a rule for every distinct sensor or parameter mentioned? "
    "(c) does each extracted rule represent a single condition at a single severity level?\n\n"
    'Reply EXCLUSIVELY with valid JSON: {"rules": [{"ruleId": "...", "class": "OperationalRule|MaintenanceRule", '
    '"station": "...", "sensor": "...", "sensorType": "...", "condition": "...", "action": "...", '
    '"severity": "MANDATORY|WARNING|HIGH|CRITICAL|MEDIUM", '
    '"critHi": "", "warnHi": "", "warnLo": "", "critLo": "", "unit": ""}, ...]}'
)

TABULAR_MINER_SYSTEM = (
    "You are a rule extractor for tabular SOPs. "
    "FIRST CHECK: does this document contain tables where each row defines a sensor or equipment item "
    "and its numeric operating limits — acceptable ranges, warning thresholds, critical thresholds? "
    "Use your understanding of industrial monitoring to identify such tables regardless of their exact "
    "column names or formatting. If no such tables exist, return {\"rules\": []} immediately.\n\n"
    "If operating-limit tables ARE present, extract one ThresholdRule for every discrete sensor or equipment item "
    "that has defined numeric limits, regardless of whether it is formatted as a table row, a list item, or prose:\n"
    "  - station: the location or station the sensor belongs to, as named in the document.\n"
    "  - sensor: compose the identifier from the location name and the measurement type, joined with an underscore. Never just the measurement type alone.\n"
    "  - sensorType: the measurement type abbreviation (e.g. TMP, PRS, HUM, FLW).\n"
    "  - critHi: the highest critical limit (numeric only, no units — strip any unit suffix).\n"
    "  - warnHi: the upper warning limit (numeric only).\n"
    "  - warnLo: the lower warning limit (numeric only).\n"
    "  - critLo: the lowest critical limit (numeric only).\n"
    "  - unit: the measurement unit (e.g. C, kPa, %, L/min, bar).\n"
    "  - action: the required response when a threshold is violated.\n"
    "  - condition: a concise description of what these thresholds govern.\n\n"
    "Also extract non-threshold rows (anomaly definitions, failure patterns, behavioral fault types): "
    "assign the class that best fits their semantic content "
    "(ThresholdRule, OperationalRule, MaintenanceRule, or AccessRule). "
    "Leave all numeric threshold fields empty for these rows.\n\n"
    "Use the document's row identifier as ruleId if present; otherwise set ruleId to ''.\n\n"
    'Reply EXCLUSIVELY with valid JSON: {"rules": [{"ruleId": "...", "class": "ThresholdRule", '
    '"station": "...", "sensor": "...", "sensorType": "...", "condition": "...", "action": "...", '
    '"severity": "CRITICAL|WARNING", '
    '"critHi": "numeric or empty", "warnHi": "numeric or empty", '
    '"warnLo": "numeric or empty", "critLo": "numeric or empty", "unit": "..."}, ...]}'
)

MATRIX_MINER_SYSTEM = (
    "You are a rule extractor for personnel access and occupancy SOPs. "
    "Extract rules ONLY from what is explicitly stated in the document.\n\n"
    "IDENTIFY AND EXTRACT from all of these source types — cover every one that is present:\n"
    "  1. Occupancy limits: for each location with a defined maximum number of people, "
    "create one AccessRule. If both warning and critical levels exist for the same location, "
    "capture both limits in a single rule's condition text.\n"
    "  2. Alarm acknowledgment requirements: for each combination of personnel role and alarm severity "
    "with a defined response time, create one AccessRule. "
    "Read role names directly from the document — do not assume any particular role naming. "
    "Extract every explicitly stated relationship between a personnel role, an alarm severity, and a required "
    "response time, regardless of whether it is formatted as a grid, list, or prose.\n"
    "  3. Explicitly labeled access rules: one AccessRule per labeled or numbered entry.\n"
    "  4. Authorization permissions: for each permission explicitly granted in an authorization table "
    "(a role or person type is allowed to access a zone or area), create one AuthorizationRule with "
    "class='AuthorizationRule' (not 'AccessRule'), "
    "condition='describe the permission using the role and location names as stated in the document', action='describe the access action as stated in the document', "
    "severity='MANDATORY', ruleId=''.\n\n"
    "DO NOT invent rules not explicitly written in the document.\n\n"
    "FIELD RULES:\n"
    "  - station: the zone or location name as written in the document. "
    "Empty string when the rule applies globally.\n"
    "  - sensor and sensorType: always empty strings for AccessRule and AuthorizationRule.\n"
    "  - critHi, warnHi, warnLo, critLo: always empty strings — numeric values belong in condition text.\n"
    "  - unit: the unit of measurement as stated in the document (e.g. persons, min, s, h). Empty string when no unit is stated.\n"
    "  - ruleId: use the document's label if present; otherwise empty string.\n\n"
    'Reply EXCLUSIVELY with valid JSON: {"rules": [{"ruleId": "...", "class": "AccessRule|AuthorizationRule", '
    '"station": "...", "sensor": "", "sensorType": "", "condition": "...", "action": "...", '
    '"severity": "MANDATORY|WARNING|CRITICAL", '
    '"critHi": "", "warnHi": "", "warnLo": "", "critLo": "", "unit": "..."}, ...]}'
)

MERGER_SYSTEM = (
    "You are a conflict resolver for extracted rules from parallel extractors. "
    "Output a single unified list.\n\n"
    "DEDUPLICATION RULE: two rules are duplicates ONLY if they share the exact same station, sensor, "
    "physical condition, threshold, and required action. "
    "Rules for different locations or equipment are NEVER duplicates even if their thresholds or actions look identical. "
    "Rules for the same sensor with different severity levels (e.g. WARNING vs CRITICAL), different thresholds, "
    "or different required actions are distinct safety constraints and must both be preserved.\n\n"
    "When a genuine duplicate exists, do not simply pick one to keep — MERGE them into a single rule "
    "that combines all non-empty fields from both instances. The merged rule must not lose any information "
    "that was present in either source.\n\n"
    "CONFLICTS: if two rules share a ruleId but describe different conditions, keep both and flag "
    "the ruleId as ambiguous — do not delete either. The downstream normalizer will resolve the IDs.\n\n"
    "Return JSON with key 'rules'."
)

# Per-doc-type completeness checklists — injected into the judge based on doc_type
_CHECKLIST_NARRATIVE = (
    "Completeness checklist for OPERATING PROCEDURE documents:\n"
    "1. EXPLICIT CONSTRAINTS: does the document state explicit operational constraints or requirements? "
    "Is there at least one extracted rule for each such constraint — regardless of how it is labeled or formatted?\n"
    "2. SENSOR AND PARAMETER COVERAGE: for each location or station, does the document mention "
    "multiple measured parameters or sensor types? Are rules present for each one that is mentioned?\n"
    "3. SEVERITY TIERS: when the document describes both a lower-severity response and a higher-severity "
    "response for the same situation, are both extracted as separate rules?\n"
    "4. CROSS-SYSTEM DEPENDENCIES: are inter-station or inter-system dependency constraints included "
    "if the document describes them?\n"
    "Only flag gaps for content actually present in the document. "
    "Do not penalise for rule types that are not in scope for this document.\n"
)

_CHECKLIST_MAINTENANCE = (
    "Completeness checklist for MAINTENANCE RULE documents:\n"
    "1. MAINTENANCE ENTRIES: is there a MaintenanceRule for every predictive-maintenance entry "
    "defined in the document? Each row or entry that defines a trigger condition and a maintenance "
    "action — regardless of how it is labeled — should correspond to one rule.\n"
    "2. CROSS-SYSTEM FAULTS: is there a MaintenanceRule for each cross-system or correlated fault "
    "pattern described in the document?\n"
    "3. FIELD QUALITY: are the location, measurement point, trigger condition, and required action "
    "all populated for each rule?\n"
    "Only flag gaps for MaintenanceRule coverage. "
    "Do not flag missing OperationalRule, ThresholdRule, or AccessRule — they are out of scope here.\n"
)

_CHECKLIST_TABULAR = (
    "Completeness checklist for THRESHOLD TABLE documents:\n"
    "1. SENSOR COVERAGE: is there a ThresholdRule for every sensor or measurement point row "
    "in the operating-limit tables? Check that no rows were skipped.\n"
    "2. BEHAVIOR DEFINITIONS: are anomaly or failure-pattern definition rows represented "
    "if the document includes them?\n"
    "3. NUMERIC ACCURACY: are the operating limit values (upper/lower critical and warning bounds) "
    "correctly extracted into the numeric threshold fields, with no unit suffixes mixed in?\n"
    "Only flag gaps for threshold and anomaly coverage. "
    "Do not penalise for missing maintenance or access rules.\n"
)

_CHECKLIST_MATRIX = (
    "Completeness checklist for ACCESS CONTROL / OCCUPANCY documents:\n"
    "1. OCCUPANCY: is there an AccessRule for every location with a defined occupancy limit?\n"
    "2. ACKNOWLEDGMENT TIMES: is there an AccessRule for every combination of personnel role and "
    "alarm severity with a defined response time? Cover every row and every severity column exhaustively.\n"
    "3. LABELED ACCESS RULES: are all explicitly labeled or numbered access rules in the document included?\n"
    "NOTE: AuthorizationRule entries from permission/authorization grids have been routed to graph_edges "
    "by the KnowledgeRouter — do not flag their absence as a gap.\n"
    "Only check for rules explicitly stated in the document. "
    "Do not flag missing threshold, maintenance, or operational rules — they are out of scope.\n"
)

_JUDGE_BASE = (
    "You are a quality judge for extracted rules. "
    "Identify gaps using these categories:\n"
    "  S1 (Class-gaps): missing rule classes that should exist in this specific SOP.\n"
    "  S2 (Coverage-gaps): specific constraints, physical locations, sensor limits, or personnel rules "
    "mentioned in the source text are absent from the extracted list. "
    "Identify the specific missing entity — do not try to count rules.\n"
    "  S3 (Quality): incomplete or ambiguous fields in existing rules (e.g. empty station, "
    "missing condition, incomplete numeric threshold values).\n"
    "  S4 (Conflicts): two rules share the same ruleId AND the same severity — this is a true "
    "conflict. Do NOT flag as S4 when two rules share a ruleId but have different severity levels "
    "(WARNING vs CRITICAL) — that is an intentional multi-severity split awaiting normalization.\n"
    "  S5 (Ungrounded): an extracted rule contains conditions, actions, or locations that are not "
    "explicitly stated in the source text — hallucinations or fabricated constraints.\n\n"
    "KEY RULE: only flag gaps for content that is actually present in the document text provided. "
    "Do not penalise for rule types that are not in scope for this document.\n\n"
)

_JUDGE_SUFFIX = (
    "\nReturn a JSON with keys:\n"
    "  'gaps': list of gap codes found (e.g. ['S1', 'S2'])\n"
    "  'suggestions': specific instructions for the retry (what to add or fix)\n"
    "  'pass': true if no critical gaps exist, false otherwise"
)

# Keep for backward compat — node_quality_gap_evaluator will build the prompt dynamically
JUDGE_SYSTEM = _JUDGE_BASE + _CHECKLIST_NARRATIVE + _JUDGE_SUFFIX

NORMALIZER_SYSTEM = (
    "You are a ruleId normalizer. Your ONLY task is to assign canonical ruleIds to all rules in the list. "
    "Do not modify the class, station, sensor, condition, action, severity, or any other field.\n\n"
    "GOLDEN RULES:\n"
    "1. DOCUMENT-SUPPLIED IDs ARE INVIOLABLE: if a rule has a non-empty ruleId that was extracted "
    "verbatim from the source document, keep it EXACTLY as-is — this overrides ALL class-conditional "
    "constraints, even if the ID pattern does not match the rule's class "
    "(e.g. an AccessRule labeled RULE-CHM01-03 in the document keeps that ID).\n"
    "2. For all other rules (empty or invented ruleId), assign the canonical ruleId using the patterns below.\n"
    "3. Ensure all ruleIds in the final list are unique. If a conflict occurs, adjust numbering.\n\n"
    "CANONICAL PATTERNS:\n"
    "   - OperationalRule: RULE-{STATION}-{NN} where {STATION} is the abbreviation from the station field "
    "(e.g., ST01, SRV01, WRH01, CHM01, RND01, CAF01). {NN} is a sequential number per station. "
    "If there are multiple OperationalRules for the same station+sensor with different severities, append 'a' for the baseline/lower severity (e.g., MANDATORY or WARNING) and 'b' for the escalated/higher severity (e.g., WARNING or CRITICAL).\n"
    "   - MaintenanceRule (simple, non‑correlated): MAINT-{NN} where {NN} is a sequential number "
    "across all simple MaintenanceRules in the full list (order as they appear).\n"
    "   - MaintenanceRule (correlated fault): RULE-CORR-{NN} (sequential, one‑based).\n"
    "   - ThresholdRule (sensor row): RULE-THR-{STATION}-{SENSOR}-{SEV} where {SENSOR} is the sensorType field "
    "(e.g., TMP, PRS, FLW, CUR, VIB, SPD, HUM, TEN, CNT) and {SEV} is WARN or CRIT based on the rule's severity field.\n"
    "   - Anomaly behavior definition (any class): RULE-ANOM-{TYPE} where {TYPE} is derived from the "
    "condition text (e.g. SPIKE, DRIFT, STUCK, OUT_OF_RANGE, CORRELATED).\n"
    "   - AccessRule (occupancy): RULE-OCC-{ZONE} where {ZONE} is derived from the zone name. "
    "For multi-word zones tied to a station, use the station prefix; for single-word zones, use the full name in uppercase. "
    "Reference mapping: 'Production Area' → PROD, 'Server Room' → SRV, "
    "'General Warehouse' → WRH, 'Chemical Storage' → CHM, 'R&D Lab' → RND, 'Cafeteria' → CAFETERIA.\n"
    "   - AccessRule (acknowledgment time): RULE-ACK-{ROLE}-{SEV} where {ROLE} is a fixed uppercase "
    "abbreviation for the role and {SEV} is exactly WARN or CRIT (never WRN, never CRT, never WARNING, never CRITICAL). "
    "Role reference: operator → OP, technician → TECH, supervisor → SUP, manager → MAN, security → SEC.\n"
    "   - AccessRule (prose, e.g. explicit RULE-ACCESS-XX): keep if already labeled; otherwise assign "
    "RULE-ACCESS-{NN} sequentially.\n\n"
    "CLASS-CONDITIONAL CONSTRAINT (applies ONLY when assigning new IDs under rule 2 — "
    "never overrides a document-supplied ID from rule 1):\n"
    "  RULE-THR-* patterns apply ONLY to class='ThresholdRule'. "
    "NEVER rename a MaintenanceRule or OperationalRule to RULE-THR-*.\n"
    "  MAINT-NN and RULE-CORR-NN apply ONLY to class='MaintenanceRule'.\n"
    "  RULE-{STATION}-{NN} applies ONLY to class='OperationalRule' when assigning a new ID.\n"
    "  RULE-OCC-*, RULE-ACK-*, RULE-ACCESS-* apply ONLY to class='AccessRule' when assigning a new ID.\n\n"
    "Do NOT change any field other than 'ruleId'. Return the full JSON list."
)



# ── VALIDATOR PROMPT (step 6 — runs before the Judge) ─────────────────────────

VALIDATOR_SYSTEM = (
    "You are a strict validator for industrial SOP rule extraction. "
    "You receive a merged set of extracted rules and the original document text. "
    "Your job is to clean the rule list BEFORE it is evaluated by the quality judge.\n\n"
    "STEP 1 — Remove false positives: delete any rule whose condition, action, or measurement value "
    "is NOT grounded in explicit text from the document. "
    "Remove rules based on assumptions about naming conventions or invented measurement points. "
    "Do NOT delete global or plant-wide rules merely because they lack a specific physical location — "
    "verify them against the document's general scope instead. "
    "Do NOT delete rules simply because they resemble another rule; rules for the same sensor with "
    "different severity tiers or thresholds are intentional and must both be preserved.\n\n"
    "STEP 2 — Add obvious false negatives: read the document as a domain expert would. "
    "Identify every explicit constraint, operational requirement, threshold definition, or maintenance "
    "trigger stated in the text — regardless of how it is labeled or formatted. "
    "If a clearly corresponding rule is absent from the extracted list, add it.\n\n"
    "STEP 3 — Fix structural errors: correct empty condition or action fields where the "
    "source text makes the value unambiguous. Do NOT invent content not in the document.\n\n"
    "Return valid JSON: {\"rules\": [ ... ]} with the validated list. "
    "Preserve all ruleIds from the input exactly — ruleId normalization is handled downstream."
)

# ── LANGGRAPH STATE ─────────────────────────────────────────────────────────────

def _merge_rules(left: list[dict], right: list[dict] | None) -> list[dict]:
    """Custom reducer: None resets, list appends. Enables parallel fan-in + clean retry resets."""
    if right is None:
        return []
    return (left or []) + right

class PipelineState(TypedDict, total=False):
    source_file:     str
    run_id:          str
    raw_text:        str
    doc_type:        str
    extracted_rules: Annotated[list[dict], _merge_rules]
    merged_rules:    list[dict]
    graph_edges:     list[dict]   # AuthorizationRule instances routed away from scoring
    final_rules:     list[dict]
    judge_feedback:  dict
    retry_count:     int

def _make_initial_state(raw_text: str, source_file: str, run_id: str) -> PipelineState:
    return PipelineState(
        source_file=source_file, run_id=run_id, raw_text=raw_text,
        doc_type="", extracted_rules=[], merged_rules=[],
        graph_edges=[], final_rules=[], judge_feedback={}, retry_count=0,
    )

def _retry_note(state: PipelineState) -> str:
    feedback    = state.get("judge_feedback", {})
    suggestions = feedback.get("suggestions", "")
    gaps        = feedback.get("gaps", [])
    if not gaps and not suggestions:
        return ""
    return (
        f"\n\nATTENTION — previous extraction was flagged (gaps: {gaps}). "
        f"{suggestions} Be thorough and extract every rule present in the document."
    )

# ── STAGE 1: CLASSIFIER ─────────────────────────────────────────────────────────

def node_document_type_classifier(state: PipelineState) -> dict:
    """Classify document type via LLM. Returns None to reset extracted_rules."""
    print("    [Classifier] Classifying document...")
    user_content = (
        f"{state['raw_text'][:4000]}\n\n"
        "---\n"
        "Now classify the document above following the STEP 1-4 rules.\n"
        "STEP 1: does the document primarily contain numeric operating limits or threshold values "
        "for sensors or equipment (upper bounds, lower bounds, warning levels, critical levels)? "
        "If yes → {\"doc_type\": \"tabular\"}.\n"
        "STEP 2: is the main content a grid or table mapping personnel roles to zones, areas, or "
        "alarm types with access permissions or required response times? "
        "If yes → {\"doc_type\": \"matrix\"}.\n"
        "STEP 3: does the document contain two or more meaningfully distinct content types that "
        "each require separate extraction — for example, numeric threshold tables alongside "
        "narrative operating procedures, or maintenance schedules alongside access rules? "
        "If distinct content types coexist and each would be missed by a single extractor → "
        "{\"doc_type\": \"mixed\"}.\n"
        "STEP 4: otherwise → {\"doc_type\": \"narrative\"}.\n"
        "Return ONLY JSON."
    )
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    raw, _ = llm_call(MODEL_CLASSIFIER, messages)
    try:
        doc_type = json.loads(raw).get("doc_type", "narrative").lower()
    except Exception:
        doc_type = "narrative"
    if doc_type not in ("tabular", "narrative", "matrix", "mixed"):
        doc_type = "narrative"
    print(f"      → {doc_type.upper()}")
    return {"doc_type": doc_type, "extracted_rules": None}  # None triggers reset via _merge_rules

# ── STAGES 2/3/4: EXTRACTORS ────────────────────────────────────────────────────

def node_narrative_miner(state: PipelineState) -> dict:
    """Extract OperationalRule and MaintenanceRule from narrative/mixed documents."""
    print("    [NarrativeMiner] Extracting narrative rules...")
    messages = [
        {"role": "system", "content": NARRATIVE_MINER_SYSTEM},
        {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT] + _retry_note(state)},
    ]
    raw, _ = llm_call(MODEL_NARRATIVE, messages)
    rules = parse_rules(raw)
    print(f"      → {len(rules)} rules")
    return {"extracted_rules": rules}

def node_tabular_miner(state: PipelineState) -> dict:
    """Extract ThresholdRule from tabular/mixed documents."""
    print("    [TabularMiner] Extracting threshold rules...")
    messages = [
        {"role": "system", "content": TABULAR_MINER_SYSTEM},
        {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT] + _retry_note(state)},
    ]
    raw, _ = llm_call(MODEL_TABULAR, messages)
    rules = parse_rules(raw)
    print(f"      → {len(rules)} rules")
    return {"extracted_rules": rules}

def node_matrix_miner(state: PipelineState) -> dict:
    """Extract AccessRule from matrix/access-control documents."""
    print("    [MatrixMiner] Extracting access rules...")
    messages = [
        {"role": "system", "content": MATRIX_MINER_SYSTEM},
        {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT] + _retry_note(state)},
    ]
    raw, _ = llm_call(MODEL_MATRIX, messages)
    rules = parse_rules(raw)
    print(f"      → {len(rules)} rules")
    return {"extracted_rules": rules}

# ── STAGE 5: MERGER ─────────────────────────────────────────────────────────────

def node_consensus_merger(state: PipelineState) -> dict:
    """LLM-based merge: resolves conflicts and deduplicates across all parallel extractors."""
    print("    [Merger] Merging...")
    all_rules = state.get("extracted_rules", [])
    doc_type  = state.get("doc_type", "narrative")

    if not all_rules:
        print("    [Merger] 0 rules — falling back to narrative extractor...")
        messages = [
            {"role": "system", "content": NARRATIVE_MINER_SYSTEM},
            {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT]},
        ]
        raw, _ = llm_call(MODEL_NARRATIVE, messages)
        fallback = parse_rules(raw)
        print(f"      → {len(fallback)} rules (fallback)")
        return {"merged_rules": fallback}

    json_in = json.dumps(all_rules, indent=2)[:14000]
    messages = [
        {"role": "system", "content": MERGER_SYSTEM},
        {"role": "user",   "content": f"Document type: {doc_type}\nRules:\n{json_in}\nReturn unified JSON."},
    ]
    raw, _ = llm_call(MODEL_ORCHESTRATOR, messages)
    merged = parse_rules(raw) or all_rules

    print(f"      → {len(merged)} rules after merge")
    return {"merged_rules": merged}

# ── STAGE 6: JUDGE ──────────────────────────────────────────────────────────────

def _judge_system_for(doc_type: str, raw_text: str) -> str:
    """Select and concatenate completeness checklists based on doc type.

    For mixed documents all relevant checklists are injected so the judge
    evaluates every extraction scope present in the document.
    """
    t = raw_text[:1000].lower()
    is_maintenance = "maint-" in t or (t.count("maintenance") >= 2 and "operating" not in t)

    if doc_type == "matrix":
        checklist = _CHECKLIST_MATRIX
    elif doc_type == "tabular":
        checklist = _CHECKLIST_TABULAR
    elif doc_type == "narrative":
        checklist = _CHECKLIST_MAINTENANCE if is_maintenance else _CHECKLIST_NARRATIVE
    elif doc_type == "mixed":
        # Inject all checklists relevant to the document's actual content
        parts = [_CHECKLIST_NARRATIVE]
        if is_maintenance:
            parts.append(_CHECKLIST_MAINTENANCE)
        parts.append(_CHECKLIST_TABULAR)
        checklist = "\n".join(parts)
    else:
        checklist = _CHECKLIST_NARRATIVE
    return _JUDGE_BASE + checklist + _JUDGE_SUFFIX


def node_quality_gap_evaluator(state: PipelineState) -> dict:
    """Quality judge: detects S1-S4 gaps, injects doc-type-specific checklist, returns feedback."""
    print("    [Judge] Assessing quality...")
    merged = state.get("merged_rules", [])
    if not merged:
        return {"judge_feedback": {"pass": True, "gaps": [], "suggestions": ""}}

    doc_type   = state.get("doc_type", "narrative")
    raw_text   = state.get("raw_text", "")
    judge_sys  = _judge_system_for(doc_type, raw_text)
    rules_json = json.dumps(merged, indent=2)[:8000]
    messages = [
        {"role": "system", "content": judge_sys},
        {"role": "user",   "content": (
            f"Document text (first 4000 chars):\n{raw_text[:4000]}\n\n"
            f"Extracted rules:\n{rules_json}"
        )},
    ]
    raw, _ = llm_call(MODEL_ORCHESTRATOR, messages)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        feedback = json.loads(cleaned)
    except Exception:
        feedback = {"pass": True, "gaps": [], "suggestions": ""}

    print(f"      → Pass: {feedback.get('pass', True)}, Gaps: {feedback.get('gaps', [])}")
    return {"judge_feedback": feedback}

# ── STAGE 7: NORMALIZER ─────────────────────────────────────────────────────────

def node_identifier_canonicalizer(state: PipelineState) -> dict:
    """Canonicalize ruleIds against known document label patterns."""
    print("    [Normalizer] Canonicalizing ruleIds...")
    merged = state.get("merged_rules", [])
    if not merged:
        return {"final_rules": []}

    json_in  = json.dumps(merged, indent=2)[:12000]
    doc_head = state.get("raw_text", "")[:2000]
    messages = [
        {"role": "system", "content": NORMALIZER_SYSTEM},
        {"role": "user",   "content": (
            f"Source document excerpt:\n{doc_head}\n\n"
            f"Rules to normalize:\n{json_in}\nReturn updated JSON."
        )},
    ]
    raw, _ = llm_call(MODEL_ORCHESTRATOR, messages)
    normalized = parse_rules(raw)
    final = normalized if normalized else merged
    print(f"      → {len(final)} rules normalized")
    return {"final_rules": final}

# ── STAGE 8: CONTENT REFINER (FP/FN scrubber) ───────────────────────────────────

def node_validate(state: PipelineState) -> dict:
    """Validator (paper step 6): pure LLM pass, runs before the Judge.

    Removes FPs not grounded in the document, adds obvious FNs,
    and fixes structurally empty fields where the source is unambiguous.
    """
    print("    [Validator] LLM validation pass...")
    rules = state.get("merged_rules", [])
    if not rules:
        return {"merged_rules": []}

    doc_text   = state.get("raw_text", "")[:SOP_TEXT_LIMIT]
    rules_json = json.dumps(rules, indent=2)[:10000]

    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM},
        {"role": "user",   "content": (
            f"Document text:\n{doc_text}\n\n"
            f"Rules to validate:\n{rules_json}\n\n"
            "Return the validated list as JSON."
        )}
    ]
    raw, _ = llm_call(MODEL_ORCHESTRATOR, messages)
    validated = parse_rules(raw)
    result = validated if validated else rules

    llm_removed = max(0, len(rules) - len(result))
    llm_added   = max(0, len(result) - len(rules))
    print(f"      → {len(result)} rules after validation (LLM: -{llm_removed} FPs +{llm_added} FNs)")
    return {"merged_rules": result}

# ── RESET NODE ──────────────────────────────────────────────────────────────────

def node_reset_retry(state: PipelineState) -> dict:
    """Increment retry counter and reset all rule accumulators for a clean re-run."""
    count = state.get("retry_count", 0) + 1
    print(f"    [Judge] Retry {count}/{MAX_JUDGE_RETRIES} triggered.")
    return {
        "retry_count":     count,
        "extracted_rules": None,  # triggers reset via _merge_rules
        "merged_rules":    [],
    }

# ── KNOWLEDGE ROUTER ────────────────────────────────────────────────────────────

def node_knowledge_router(state: PipelineState) -> dict:
    """Deterministic semantic router that runs after ConsensusMerger.

    Separates extracted knowledge into two streams:
      - merged_rules  : rule nodes that flow to QualityGapEvaluator and scoring
      - graph_edges   : AuthorizationRule instances (role×zone YES/NO grid) that
                        represent binary graph relationships (authorized_for edges
                        in Neo4j) rather than independently actionable rule nodes.

    Rules with a non-standard class are passed through unchanged so the
    Normalizer and Judge can handle them without hardcoded assumptions.
"""
    merged = state.get("merged_rules", [])

    _SCORING_CLASSES = {"OperationalRule", "MaintenanceRule", "ThresholdRule", "AccessRule"}

    rule_nodes: list[dict] = []
    graph_edges: list[dict] = []

    for r in merged:
        cls = r.get("class", "")
        if cls == "AuthorizationRule":
            # Binary permission from the role×zone grid → Neo4j authorized_for edge
            graph_edges.append({**r, "_edge_type": "authorized_for"})
        elif cls not in _SCORING_CLASSES:
            rule_nodes.append(r)
        else:
            rule_nodes.append(r)

    print(
        f"    [KnowledgeRouter] {len(rule_nodes)} rule_nodes → QualityGapEvaluator | "
        f"{len(graph_edges)} graph_edges → Neo4j (authorized_for)"
    )
    return {"merged_rules": rule_nodes, "graph_edges": graph_edges}

# ── CONDITIONAL EDGES ───────────────────────────────────────────────────────────

def route_by_doc_type(state: PipelineState) -> List[str]:
    """Fan-out to the correct extractor(s) based on document type."""
    dt = state.get("doc_type", "narrative")
    if dt == "tabular":  return ["TabularThresholdMiner"]
    if dt == "narrative": return ["NarrativeRuleMiner"]
    if dt == "matrix":   return ["AccessMatrixMiner"]
    if dt == "mixed":    return ["NarrativeRuleMiner", "TabularThresholdMiner"]
    return ["NarrativeRuleMiner"]

def route_after_judge(state: PipelineState) -> str:
    if state.get("judge_feedback", {}).get("pass", True):
        return "IdentifierCanonicalizer"
    if state.get("retry_count", 0) < MAX_JUDGE_RETRIES:
        return "ResetAndRetry"
    print("    [Judge] Max retries reached — proceeding to normalization.")
    return "IdentifierCanonicalizer"

# ── BUILD GRAPH ─────────────────────────────────────────────────────────────────

def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("DocumentTypeClassifier",    node_document_type_classifier)
    graph.add_node("NarrativeRuleMiner",        node_narrative_miner)
    graph.add_node("TabularThresholdMiner",     node_tabular_miner)
    graph.add_node("AccessMatrixMiner",         node_matrix_miner)
    graph.add_node("ConsensusMerger",           node_consensus_merger)
    graph.add_node("KnowledgeRouter",           node_knowledge_router)
    graph.add_node("Validator",                 node_validate)
    graph.add_node("QualityGapEvaluator",       node_quality_gap_evaluator)
    graph.add_node("IdentifierCanonicalizer",   node_identifier_canonicalizer)
    graph.add_node("ResetAndRetry",             node_reset_retry)

    graph.add_edge(START, "DocumentTypeClassifier")

    graph.add_conditional_edges(
        "DocumentTypeClassifier", route_by_doc_type,
        ["NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"],
    )

    graph.add_edge("NarrativeRuleMiner",    "ConsensusMerger")
    graph.add_edge("TabularThresholdMiner", "ConsensusMerger")
    graph.add_edge("AccessMatrixMiner",     "ConsensusMerger")

    graph.add_edge("ConsensusMerger",  "KnowledgeRouter")
    graph.add_edge("KnowledgeRouter",  "Validator")
    graph.add_edge("Validator",        "QualityGapEvaluator")

    graph.add_conditional_edges(
        "QualityGapEvaluator", route_after_judge,
        ["IdentifierCanonicalizer", "ResetAndRetry"],
    )
    graph.add_edge("ResetAndRetry",           "DocumentTypeClassifier")
    graph.add_edge("IdentifierCanonicalizer", END)

    return graph.compile()

# ── ABLATION STUDY ──────────────────────────────────────────────────────────────

ABLATION_OPTIONS = (
    "no_consensus_merger",
    "no_quality_gap_evaluator",
    "no_identifier_canonicalizer",
    "no_validator",
)

_ABLATION_TAGS = {
    "no_consensus_merger":         "noCM",
    "no_quality_gap_evaluator":    "noQGE",
    "no_identifier_canonicalizer": "noIC",
    "no_validator":                "noV",
}


def node_simple_passthrough_merge(state: PipelineState) -> dict:
    """Ablation passthrough: deterministic dedup of extracted_rules → merged_rules, no LLM."""
    all_rules = state.get("extracted_rules", [])
    if not all_rules:
        print("    [SimplePassthrough] 0 rules — fallback to narrative extractor...")
        messages = [
            {"role": "system", "content": NARRATIVE_MINER_SYSTEM},
            {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT]},
        ]
        raw, _ = llm_call(MODEL_NARRATIVE, messages)
        fallback = parse_rules(raw)
        print(f"      → {len(fallback)} rules (fallback)")
        return {"merged_rules": fallback}

    seen_content: set[tuple] = set()
    deduped: list[dict] = []
    for r in all_rules:
        ck = (r.get("station", "").strip(),
              r.get("sensorType", "").strip(),
              r.get("condition", "").strip()[:60])
        if all(ck) and ck in seen_content:
            continue
        seen_content.add(ck)
        deduped.append(r)
    print(f"    [SimplePassthrough] {len(deduped)} rules after dedup (no LLM merge)")
    return {"merged_rules": deduped}


def node_passthrough_normalize(state: PipelineState) -> dict:
    """Ablation passthrough: copy merged_rules → final_rules without LLM normalization."""
    merged = state.get("merged_rules", [])
    print(f"    [PassthroughNormalize] {len(merged)} rules passed through (no IdentifierCanonicalizer)")
    return {"final_rules": merged}


def _route_no_canonicalizer(state: PipelineState) -> str:
    """route_after_judge variant that targets PassthroughNormalize instead of IdentifierCanonicalizer."""
    if state.get("judge_feedback", {}).get("pass", True):
        return "PassthroughNormalize"
    if state.get("retry_count", 0) < MAX_JUDGE_RETRIES:
        return "ResetAndRetry"
    print("    [Judge] Max retries reached — proceeding without normalizer.")
    return "PassthroughNormalize"


def build_ablation_pipeline(ablation: str):
    """Build a pipeline with exactly one component removed for ablation study.

    KnowledgeRouter is NOT an ablation variable — it is present in all variants
    because it performs a correctness transformation (routing graph edges away from
    scoring), not an optional quality-improvement step.

    ablation must be one of ABLATION_OPTIONS:
      no_consensus_merger         — skip ConsensusMerger (deterministic dedup only)
      no_quality_gap_evaluator    — skip judge loop entirely (no retries)
      no_identifier_canonicalizer — skip IdentifierCanonicalizer (pass merged_rules through)
      no_validator                — skip Validator (KnowledgeRouter → Judge directly)
    """
    if ablation not in ABLATION_OPTIONS:
        raise ValueError(f"Unknown ablation {ablation!r}. Choose from: {ABLATION_OPTIONS}")

    graph = StateGraph(PipelineState)

    if ablation == "no_consensus_merger":
        graph.add_node("DocumentTypeClassifier",  node_document_type_classifier)
        graph.add_node("NarrativeRuleMiner",      node_narrative_miner)
        graph.add_node("TabularThresholdMiner",   node_tabular_miner)
        graph.add_node("AccessMatrixMiner",       node_matrix_miner)
        graph.add_node("SimplePassthroughMerge",  node_simple_passthrough_merge)
        graph.add_node("KnowledgeRouter",         node_knowledge_router)
        graph.add_node("Validator",               node_validate)
        graph.add_node("QualityGapEvaluator",     node_quality_gap_evaluator)
        graph.add_node("IdentifierCanonicalizer", node_identifier_canonicalizer)
        graph.add_node("ResetAndRetry",           node_reset_retry)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_conditional_edges("DocumentTypeClassifier", route_by_doc_type,
            ["NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"])
        for _m in ("NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"):
            graph.add_edge(_m, "SimplePassthroughMerge")
        graph.add_edge("SimplePassthroughMerge", "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter",        "Validator")
        graph.add_edge("Validator",              "QualityGapEvaluator")
        graph.add_conditional_edges("QualityGapEvaluator", route_after_judge,
            ["IdentifierCanonicalizer", "ResetAndRetry"])
        graph.add_edge("ResetAndRetry",           "DocumentTypeClassifier")
        graph.add_edge("IdentifierCanonicalizer", END)

    elif ablation == "no_quality_gap_evaluator":
        graph.add_node("DocumentTypeClassifier",  node_document_type_classifier)
        graph.add_node("NarrativeRuleMiner",      node_narrative_miner)
        graph.add_node("TabularThresholdMiner",   node_tabular_miner)
        graph.add_node("AccessMatrixMiner",       node_matrix_miner)
        graph.add_node("ConsensusMerger",         node_consensus_merger)
        graph.add_node("KnowledgeRouter",         node_knowledge_router)
        graph.add_node("Validator",               node_validate)
        graph.add_node("IdentifierCanonicalizer", node_identifier_canonicalizer)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_conditional_edges("DocumentTypeClassifier", route_by_doc_type,
            ["NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"])
        for _m in ("NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"):
            graph.add_edge(_m, "ConsensusMerger")
        graph.add_edge("ConsensusMerger",         "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter",         "Validator")
        graph.add_edge("Validator",               "IdentifierCanonicalizer")
        graph.add_edge("IdentifierCanonicalizer", END)

    elif ablation == "no_identifier_canonicalizer":
        graph.add_node("DocumentTypeClassifier", node_document_type_classifier)
        graph.add_node("NarrativeRuleMiner",     node_narrative_miner)
        graph.add_node("TabularThresholdMiner",  node_tabular_miner)
        graph.add_node("AccessMatrixMiner",      node_matrix_miner)
        graph.add_node("ConsensusMerger",        node_consensus_merger)
        graph.add_node("KnowledgeRouter",        node_knowledge_router)
        graph.add_node("Validator",              node_validate)
        graph.add_node("QualityGapEvaluator",    node_quality_gap_evaluator)
        graph.add_node("PassthroughNormalize",   node_passthrough_normalize)
        graph.add_node("ResetAndRetry",          node_reset_retry)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_conditional_edges("DocumentTypeClassifier", route_by_doc_type,
            ["NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"])
        for _m in ("NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"):
            graph.add_edge(_m, "ConsensusMerger")
        graph.add_edge("ConsensusMerger", "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter", "Validator")
        graph.add_edge("Validator",       "QualityGapEvaluator")
        graph.add_conditional_edges("QualityGapEvaluator", _route_no_canonicalizer,
            ["PassthroughNormalize", "ResetAndRetry"])
        graph.add_edge("ResetAndRetry",        "DocumentTypeClassifier")
        graph.add_edge("PassthroughNormalize", END)

    elif ablation == "no_validator":
        graph.add_node("DocumentTypeClassifier",  node_document_type_classifier)
        graph.add_node("NarrativeRuleMiner",      node_narrative_miner)
        graph.add_node("TabularThresholdMiner",   node_tabular_miner)
        graph.add_node("AccessMatrixMiner",       node_matrix_miner)
        graph.add_node("ConsensusMerger",         node_consensus_merger)
        graph.add_node("KnowledgeRouter",         node_knowledge_router)
        graph.add_node("QualityGapEvaluator",     node_quality_gap_evaluator)
        graph.add_node("IdentifierCanonicalizer", node_identifier_canonicalizer)
        graph.add_node("ResetAndRetry",           node_reset_retry)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_conditional_edges("DocumentTypeClassifier", route_by_doc_type,
            ["NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"])
        for _m in ("NarrativeRuleMiner", "TabularThresholdMiner", "AccessMatrixMiner"):
            graph.add_edge(_m, "ConsensusMerger")
        graph.add_edge("ConsensusMerger", "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter", "QualityGapEvaluator")
        graph.add_conditional_edges("QualityGapEvaluator", route_after_judge,
            ["IdentifierCanonicalizer", "ResetAndRetry"])
        graph.add_edge("ResetAndRetry",           "DocumentTypeClassifier")
        graph.add_edge("IdentifierCanonicalizer", END)

    return graph.compile()

# ── RESULT FIELDS ───────────────────────────────────────────────────────────────

_RULE_FIELDS = [
    "ruleId", "class", "station", "sensor", "sensorType",
    "condition", "action", "severity",
    "critHi", "warnHi", "warnLo", "critLo", "unit",
    "source_file", "run_id", "doc_type_classified",
]

_GRAPH_EDGE_FIELDS = [
    "class", "_edge_type", "station", "condition", "action", "severity",
    "source_file", "run_id",
]

_METADATA_FIELDS = [
    "run_id", "source_file", "doc_type", "total_rules", "graph_edges",
    "retry_count", "duration_sec", "prompt_tokens", "completion_tokens", "total_tokens",
]

# ── SAVING ──────────────────────────────────────────────────────────────────────

def _save_results(run_id: str, filename: str, rules: list[dict], doc_type: str) -> None:
    if not rules:
        print(f"    No rules to save for {filename}")
        return
    extra      = sorted({k for r in rules for k in r.keys()} - set(_RULE_FIELDS))
    fieldnames = _RULE_FIELDS + extra
    for r in rules:
        for f in _RULE_FIELDS: r.setdefault(f, "")
        r["source_file"]        = filename
        r["run_id"]             = run_id
        r["doc_type_classified"] = doc_type
    stem = os.path.splitext(filename)[0]
    out  = os.path.join(RESULTS_DIR, f"agentic_{stem}_{run_id}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rules)
    print(f"    [Saved] {len(rules)} rules → {out}")

def _save_graph_edges(run_id: str, filename: str, edges: list[dict]) -> None:
    """Save AuthorizationRule graph edges to a separate CSV for the Neo4j loader."""
    if not edges:
        return
    extra      = sorted({k for e in edges for k in e.keys()} - set(_GRAPH_EDGE_FIELDS))
    fieldnames = _GRAPH_EDGE_FIELDS + extra
    for e in edges:
        for f in _GRAPH_EDGE_FIELDS: e.setdefault(f, "")
        e["source_file"] = filename
        e["run_id"]      = run_id
    stem = os.path.splitext(filename)[0]
    out  = os.path.join(RESULTS_DIR, f"edges_{stem}_{run_id}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(edges)
    print(f"    [Edges] {len(edges)} graph edges → {out}")

def _append_metadata(record: dict) -> None:
    write_header = not os.path.exists(METADATA_FILE)
    with open(METADATA_FILE, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METADATA_FIELDS, extrasaction="ignore")
        if write_header: w.writeheader()
        w.writerow(record)

# ── EXECUTION ───────────────────────────────────────────────────────────────────

def run_extraction(force_redo: bool = True, n_runs: int = 1, seed: int = SEED,
                   ablation: str | None = None) -> None:
    global _registry_ref, SEED
    SEED = seed

    print(f"\n{'='*65}")
    print(f"  Enhanced Agentic SOP Extraction Pipeline")
    print(f"  CLASSIFY={MODEL_CLASSIFIER}  NARRATIVE={MODEL_NARRATIVE}")
    print(f"  TABULAR={MODEL_TABULAR}  MATRIX={MODEL_MATRIX}")
    print(f"  ORCHESTRATOR={MODEL_ORCHESTRATOR}")
    print(f"  Seed={seed} | Runs={n_runs} | MaxJudgeRetries={MAX_JUDGE_RETRIES}")
    if ablation:
        print(f"  ABLATION: {ablation.upper()} (tag={_ABLATION_TAGS[ablation]})")
    print(f"{'='*65}")

    abs_texts = os.path.abspath(TEXTS_DIR)
    if not os.path.isdir(abs_texts):
        print(f"Error: texts dir not found: {abs_texts}"); return
    txt_files = sorted(f for f in os.listdir(abs_texts) if f.endswith(".txt"))
    if not txt_files:
        print(f"Error: no .txt files in {abs_texts}"); return
    print(f"  SOP files: {txt_files}\n")

    pipeline = build_ablation_pipeline(ablation) if ablation else build_pipeline()
    registry = {} if force_redo else _load_registry()
    _registry_ref = registry

    for run_n in range(1, n_runs + 1):
        _abl_prefix = f"abl_{_ABLATION_TAGS[ablation]}_" if ablation else "agentic_"
        run_id = f"{_abl_prefix}s{seed}_run{run_n}"
        _heartbeat.set_experiment(run_id)

        if not force_redo and _is_completed(registry, run_id):
            print(f"  SKIP {run_id}"); continue

        print(f"\n{'─'*65}")
        print(f"  RUN {run_n}/{n_runs} | {run_id}")
        print(f"{'─'*65}")

        all_rules: list[dict] = []
        all_edges: list[dict] = []
        errors    = 0
        t_start   = time.time()

        for filename in txt_files:
            filepath = os.path.join(abs_texts, filename)
            print(f"\n  >> Processing: {filename}")
            _reset_tokens()
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                initial_state = _make_initial_state(raw_text, filename, run_id)
                t_file        = time.time()
                result        = pipeline.invoke(initial_state)
                file_duration = time.time() - t_file

                final_rules  = result.get("final_rules", [])
                graph_edges  = result.get("graph_edges", [])
                doc_type     = result.get("doc_type", "")
                retries      = result.get("retry_count", 0)

                print(
                    f"    {len(final_rules)} rules | {len(graph_edges)} graph edges | "
                    f"doc_type={doc_type} | retries={retries} | {file_duration:.1f}s"
                )

                _save_results(run_id, filename, final_rules, doc_type)
                _save_graph_edges(run_id, filename, graph_edges)
                tokens = _get_tokens()
                _append_metadata({
                    "run_id":             run_id,
                    "source_file":        filename,
                    "doc_type":           doc_type,
                    "total_rules":        len(final_rules),
                    "graph_edges":        len(graph_edges),
                    "retry_count":        retries,
                    "duration_sec":       round(file_duration, 2),
                    "prompt_tokens":      tokens.prompt_tokens,
                    "completion_tokens":  tokens.completion_tokens,
                    "total_tokens":       tokens.total_tokens,
                })
                all_rules.extend(final_rules)
                all_edges.extend(graph_edges)

            except Exception as e:
                errors += 1
                print(f"    ERROR in {filename}: {e}")

        duration = time.time() - t_start
        print(f"\n  Run {run_id}: {len(all_rules)} rules | {len(all_edges)} graph edges | {errors} errors | {duration:.0f}s")
        if all_rules:
            # Save combined rule output for evaluation (all SOPs in one file)
            _file_prefix = f"abl_{_ABLATION_TAGS[ablation]}" if ablation else "ext_multi_agent"
            combined_name = f"{_file_prefix}_s{seed}_run{run_n}.csv"
            combined_path = os.path.join(RESULTS_DIR, combined_name)
            extra      = sorted({k for r in all_rules for k in r.keys()} - set(_RULE_FIELDS))
            fieldnames = _RULE_FIELDS + extra
            with open(combined_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_rules)
            print(f"  [Combined] {len(all_rules)} rules → {combined_path}")

            # Save combined graph edges (authorization relationships for Neo4j loader)
            if all_edges:
                edges_name = f"{_file_prefix}_edges_s{seed}_run{run_n}.csv"
                edges_path = os.path.join(RESULTS_DIR, edges_name)
                extra_e    = sorted({k for e in all_edges for k in e.keys()} - set(_GRAPH_EDGE_FIELDS))
                fn_e       = _GRAPH_EDGE_FIELDS + extra_e
                with open(edges_path, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fn_e, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(all_edges)
                print(f"  [Edges]    {len(all_edges)} graph edges → {edges_path}")

            _mark_completed(registry, run_id)
        else:
            print("    Warning: 0 rules extracted — will retry next time.")

    _heartbeat.set_experiment("completed")
    completed = sum(1 for v in registry.values() if v == "completed")
    print(f"\n{'='*65}")
    print(f"Pipeline complete: {completed}/{n_runs} runs")
    print(f"  Results:  {RESULTS_DIR}/")
    print(f"  Metadata: {METADATA_FILE}")

# ── CLI ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Agentic SOP Rule Extraction")
    parser.add_argument("--runs",     type=int, default=1,     help="Runs per seed")
    parser.add_argument("--no-force", dest="no_force", action="store_true",
                        help="Skip runs already marked completed in the registry")
    parser.add_argument("--seeds",    nargs="+", type=int, default=[SEED],
                        help=f"RNG seeds to run in sequence (default: [{SEED}])")
    parser.add_argument("--ablation", choices=ABLATION_OPTIONS, default=None,
                        help="Remove one pipeline component for ablation study")
    args = parser.parse_args()

    for seed_val in args.seeds:
        print(f"\n{'='*65}")
        print(f"  SEED = {seed_val}  ({args.seeds.index(seed_val)+1}/{len(args.seeds)})")
        print(f"{'='*65}")
        run_extraction(
            force_redo=not args.no_force,
            n_runs=args.runs,
            seed=seed_val,
            ablation=args.ablation,
        )