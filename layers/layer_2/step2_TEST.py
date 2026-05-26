"""
step2_ma.py
==================================================
Enhanced LangGraph multi-agent SOP extraction pipeline.

Combines the clean fan-in architecture (Annotated state + custom reducer)
from agentic_sop_extraction_pipeline.py with a single, semantic Universal
Rule Miner that replaces the previous layout-specific miners and document
type router.  The new miner uses chain-of-thought reasoning to extract
all rule types (thresholds, operational, access, etc.) from any document
format – table, paragraph, or matrix – without needing to know the layout
in advance.

The document type classifier is retained solely to feed the quality judge
with the appropriate completeness checklist (narrative, tabular, matrix,
or mixed).  It does *not* control which miner runs.

Architecture (simplified):
    DocumentTypeClassifier  (LLM classification for judge checklists)
      ↓
    UniversalRuleMiner      (semantic + CoT, extracts all rule classes)
      ↓
    ConsensusMerger         (LLM dedup/merge – removed in no_consensus_merger ablation)
      ↓
    KnowledgeRouter         (routes AuthorizationRule → graph_edges, others → merged_rules)
      ↓
    Validator               (pre‑judge FP/FN scrubber – removed in no_validator ablation)
      ↓
    QualityGapEvaluator     (judge loop, max MAX_JUDGE_RETRIES – removed in no_quality_gap_evaluator ablation)
      ↓
    IdentifierCanonicalizer (ruleId normalization – replaced by passthrough in no_identifier_canonicalizer ablation)

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
      Pipeline: Classifier → UniversalMiner → SimplePassthroughMerge
                → KnowledgeRouter → Validator → QualityGapEvaluator
                → IdentifierCanonicalizer
      Output  : abl_noCM_s<seed>_run<n>.csv
      Use when: measuring the contribution of LLM-based conflict
                resolution vs. rule-based deduplication alone.

  --ablation no_quality_gap_evaluator
      Removes : QualityGapEvaluator (judge loop + retries)
      Pipeline: Classifier → UniversalMiner → ConsensusMerger
                → KnowledgeRouter → Validator → IdentifierCanonicalizer
      Output  : abl_noQGE_s<seed>_run<n>.csv
      Use when: measuring how much the iterative judge loop improves
                recall (no retries, no gap feedback).

  --ablation no_identifier_canonicalizer
      Removes : IdentifierCanonicalizer (LLM ruleId normalization)
      Replaces: PassthroughNormalize (copies merged_rules → final_rules)
      Pipeline: Classifier → UniversalMiner → ConsensusMerger
                → KnowledgeRouter → Validator → QualityGapEvaluator
                → PassthroughNormalize
      Output  : abl_noIC_s<seed>_run<n>.csv
      Use when: measuring the impact of canonical ruleId assignment
                on downstream evaluation / F1 scoring.

  --ablation no_validator
      Removes : Validator (pre-judge FP/FN scrubber)
      Pipeline: Classifier → UniversalMiner → ConsensusMerger
                → KnowledgeRouter → QualityGapEvaluator
                → IdentifierCanonicalizer
      Output  : abl_noV_s<seed>_run<n>.csv
      Use when: measuring how many false positives / false negatives
                the validator catches before the judge loop runs.

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
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv(dotenv_path=os.path.join(os.
path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

# ── PATHS ──────────────────────────────────────────────────────────────────────

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TEXTS_DIR      = os.path.join(_SCRIPT_DIR, "..", "texts")
RESULTS_DIR    = os.path.join(_SCRIPT_DIR, "agentic_results")
STATE_DIR      = os.path.join(_SCRIPT_DIR, "..", "state")
REGISTRY_FILE  = os.path.join(STATE_DIR, "agentic_registry.json")
METADATA_FILE  = os.path.join(STATE_DIR, "agentic_metadata.csv")
HEARTBEAT_FILE = os.path.join(_SCRIPT_DIR, "..", "heartbeat.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# ── PER-NODE MODEL ASSIGNMENTS ────────────────────────────────────────────────

MODEL_NODE_CLASSIFIER    = os.environ.get("MODEL_NODE_CLASSIFIER",    "qwen/qwen3-4b-2507")
MODEL_NODE_MINER         = os.environ.get("MODEL_NODE_MINER",         "ministral-3:8b")
MODEL_NODE_MERGER        = os.environ.get("MODEL_NODE_MERGER",        "ministral-3:14b")
MODEL_NODE_EVALUATOR     = os.environ.get("MODEL_NODE_EVALUATOR",     "ministral-3:14b")
MODEL_NODE_CANONICALIZER = os.environ.get("MODEL_NODE_CANONICALIZER", "ministral-3:14b")
MODEL_NODE_VALIDATOR     = os.environ.get("MODEL_NODE_VALIDATOR",     "gemma3:12b")
MODEL_NODE_PASSTHROUGH   = os.environ.get("MODEL_NODE_PASSTHROUGH",   "ministral-3:8b")

# ── LLM PARAMETERS ────────────────────────────────────────────────────────────

LLM_TIMEOUT_SEC   = 600    # local Ollama/LMStudio models are slow under heavy context; 10 min avoids premature failures
LLM_NUM_CTX       = 16384  # KV-cache slots per Ollama request; must exceed the actual prompt length or the model truncates silently
MAX_OUTPUT_TOKENS = 16384
MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 15.0
SEED              = 42
MAX_JUDGE_RETRIES = 2      # each retry is a full re-extraction; 2 retries balance recall improvement against latency cost
SOP_TEXT_LIMIT    = 8000   # leaves headroom for the system prompt and output tokens within the model's context window

# ── CONNECTIONS ────────────────────────────────────────────────────────────────

_OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL",   "http://localhost:11434/v1")
_OLLAMA_API_KEY    = os.environ.get("OLLAMA_API_KEY",    "ollama")
_LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
_LMSTUDIO_API_KEY  = os.environ.get("LMSTUDIO_API_KEY",  "lm-studio")
LMSTUDIO_MODELS    = set(m.strip() for m in os.environ.get("LMSTUDIO_MODELS", "").split(",") if m.strip())

# Ministral's chat template has no system turn; prepending system content to the first user message
# avoids a template rendering error at the inference server.
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

# Thread-local storage isolates per-document token counts; safe if files are ever processed concurrently.
_tls = threading.local()
def _reset_tokens() -> None:  _tls.acc = LLMUsage()
def _get_tokens()  -> LLMUsage: return getattr(_tls, "acc", LLMUsage())
def _add_tokens(u: LLMUsage) -> None: _tls.acc = getattr(_tls, "acc", LLMUsage()) + u

# ── HEARTBEAT ──────────────────────────────────────────────────────────────────

class HeartbeatThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)  # daemon=True: dies automatically with the main process, no orphaned writer after SIGTERM
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

# Shared with the SIGINT/SIGTERM handler so partial registry state can be flushed on interrupt
# without threading it through every call frame.
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
    os.replace(tmp, REGISTRY_FILE)  # atomic on POSIX: prevents a half-written file if the process is killed mid-write

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
    """Parse LLM JSON output with a cascade of repair strategies.

    Models sometimes truncate mid-JSON or wrap output in markdown fences.
    Each fallback targets a distinct failure mode:
      1. Direct parse       — output is well-formed JSON.
      2. Bracket repair     — model ran out of tokens; rebalance brackets and retry.
      3. Fence/pattern scan — output is wrapped in a code block or an outer dict.
      4. ruleId salvage     — malformed output; extract complete rule objects by scanning.
    Reasoning traces (<think>…</think>) from CoT models are stripped before any attempt.
    """
    if not raw: return []
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    def _extract(text: str) -> list[dict]:
        data = json.loads(text)
        # Support both { "reasoning": "...", "rules": [...] } and plain lists
        if isinstance(data, dict):
            reasoning = data.get("reasoning", "")
            if reasoning:
                print(f"      [LLM reasoning] {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
            rules = data.get("rules", data)  # fallback to whole object if no 'rules' key
        else:
            rules = data
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

# The classifier's output is used exclusively to select the judge's completeness checklist
# (tabular/matrix/narrative/mixed coverage criteria). It does NOT select which miner runs —
# UniversalRuleMiner always handles all layouts semantically. Separating classification from
# routing means the judge applies the right evaluation criteria without gating extraction quality
# behind a classification error.
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

# ── UNIVERSAL RULE MINER (replaces all layout-specific miners) ─────────────────
UNIVERSAL_MINER_SYSTEM = (
    "You are a Precision Data Engineer specializing in SCADA systems and industrial SOPs.\n"
    "Your task is to extract operational, maintenance, threshold, and access rules from raw, unstructured industrial documents.\n\n"
    "Do not rely on any specific layout, table format, or naming convention in the source text. "
    "Focus purely on the semantic meaning to identify rules and constraints.\n\n"
    "FIELD DEFINITIONS FOR EXTRACTION:\n"
    "- ruleId: The explicit code/label given in the text. Leave empty (\"\") if none exists.\n"
    "- class: Must be exactly one of [ThresholdRule, OperationalRule, MaintenanceRule, AccessRule]. "
    "ThresholdRule: defines sensor measurement alarm boundaries (numeric limits for a physical sensor). "
    "OperationalRule: describes what must happen when a sensor triggers a response (condition → action). "
    "MaintenanceRule: describes maintenance actions, drift-based conditions, or causal links between stations. "
    "AccessRule: governs personnel authorization, zone occupancy, or role-based obligations (e.g. acknowledgment time). "
    "Assign AccessRule even when numeric limits are present, if the rule semantics concern personnel or zone access rather than sensor monitoring.\n"
    "- station: The physical location or zone identifier as it appears in the document. "
    "Preserve the original identifier — do not invent or normalize it.\n"
    "- sensor: Compose using the station identifier and sensor type abbreviation separated by an underscore, "
    "following the document's naming convention. Empty if not applicable.\n"
    "- sensorType: The measurement type code from the ontology vocabulary "
    "(TMP, PRS, FLW, CUR, VIB, SPD, HUM, TEN, CNT). Empty if not applicable.\n"
    "- condition: A concise description of the triggering condition or limit.\n"
    "- action: The required response or corrective action.\n"
    "- severity: Infer from context. Must be one of [CRITICAL, WARNING, MANDATORY, HIGH, MEDIUM, LOW].\n"
    "- unit: The measurement unit (e.g., C, bar, L/min, A, mm/s). Empty if none.\n\n"
    "Numeric Threshold Fields (For ThresholdRule only. Must be NUMERIC ONLY, no unit suffixes. "
    "Leave empty for non-threshold rules):\n"
    "- critHi: The highest critical limit.\n"
    "- warnHi: The upper warning limit.\n"
    "- warnLo: The lower warning limit.\n"
    "- critLo: The lowest critical limit.\n"
    "Note: ThresholdRules also include statistical anomaly detection patterns (e.g. transient spikes, "
    "monotonic drift, stuck/frozen readings, sustained out-of-range conditions). These may have no "
    "numeric boundary values and no station — they define detection logic applicable across sensors. "
    "Leave all four numeric fields empty for such rules.\n\n"
    "INSTRUCTIONS:\n"
    "1. Extract every explicit constraint, limit, or procedure. "
    "Include cross-station fault chains and correlated failure patterns as MaintenanceRules — "
    "these describe causal links between stations and may appear in any format (table, prose, or list).\n"
    "2. Multi-severity handling: for ThresholdRule, if warning and critical bounds apply to the same "
    "sensor measurement, populate all four numeric fields (critHi, warnHi, warnLo, critLo) in a single "
    "rule rather than creating separate rules per severity tier. For OperationalRule, MaintenanceRule, "
    "and AccessRule, if the required action differs between severity tiers, extract each as a separate "
    "rule with the appropriate severity field.\n"
    "3. Base your extraction solely on the provided text. Do not invent rules.\n\n"
    "OUTPUT FORMAT:\n"
    "Reply EXCLUSIVELY with a JSON object containing two keys:\n"
    "- \"reasoning\": A brief string explaining your thought process for untangling the text.\n"
    "- \"rules\": A list of objects adhering exactly to the fields defined above.\n"
)

CONFLICT_ADJUDICATOR_SYSTEM = (
    "You are a conflict resolver for extracted industrial SOP rules. "
    "You receive a list of conflict groups. Each group contains 2 or more rule candidates "
    "that share the same class, station, and sensor type.\n\n"
    "For each group, decide:\n"
    "  MERGE — if the candidates describe the same physical constraint with the same required action "
    "(minor wording differences only). Output a single merged rule combining all non-empty fields.\n"
    "  KEEP_ALL — if the required actions differ between candidates, or the conditions describe "
    "genuinely different situations. Keep all candidates as separate rules.\n\n"
    "Never merge rules whose required action differs, even if station and sensor match.\n"
    "Never add fields not present in any of the input candidates.\n\n"
    "Return ONLY valid JSON: {\"resolved\": [<flat list of output rules>]}"
)

# ── MERGER HELPERS ──────────────────────────────────────────────────────────────

_ALLOWED_CLASSES    = {"ThresholdRule", "OperationalRule", "MaintenanceRule", "AccessRule"}
_ALLOWED_SEVERITIES = {"CRITICAL", "WARNING", "MANDATORY", "HIGH", "MEDIUM", "LOW"}

def _n_nonempty(rule: dict) -> int:
    return sum(1 for v in rule.values() if v not in (None, "", []))

def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().strip())[:80]

def _has_crit_fields(r: dict) -> bool:
    return bool(r.get("critHi") or r.get("critLo"))

def _has_warn_fields(r: dict) -> bool:
    return bool(r.get("warnHi") or r.get("warnLo"))

def _has_all_four(r: dict) -> bool:
    return all(r.get(f) for f in ("critHi", "warnHi", "warnLo", "critLo"))

def _guard_fields(rule: dict, original_candidates: list[dict]) -> dict:
    """Revert categorical fields corrupted by LLM to values from the richest input candidate."""
    best = max(original_candidates, key=_n_nonempty)
    out = dict(rule)
    if out.get("class") not in _ALLOWED_CLASSES:
        out["class"] = best.get("class", "")
    if out.get("severity", "").upper() not in _ALLOWED_SEVERITIES:
        out["severity"] = best.get("severity", "")
    if not out.get("station") and best.get("station"):
        out["station"] = best["station"]
    return out

def _deterministic_post_filter(rules: list[dict]) -> list[dict]:
    """Deterministic cleanup applied after every LLM pass.

    1. Class validation — drop rules with an invalid class value.
    2. Numeric completeness — ThresholdRules with no threshold fields at all are dropped.
    3. Duplicate elimination — keep the richest rule per content signature.
    """
    seen_sigs: dict[str, dict] = {}
    for r in rules:
        if r.get("class") not in _ALLOWED_CLASSES:
            continue
        if r.get("class") == "ThresholdRule":
            if not any(r.get(f) for f in ("critHi", "warnHi", "warnLo", "critLo")):
                continue
        sig = "|".join([
            r.get("class", "").lower(),
            r.get("station", "").lower(),
            r.get("sensorType", "").lower(),
            r.get("severity", "").lower(),
            _norm_text(r.get("condition", "")),
            _norm_text(r.get("action", "")),
        ])
        if sig not in seen_sigs or _n_nonempty(r) > _n_nonempty(seen_sigs[sig]):
            seen_sigs[sig] = r
    return list(seen_sigs.values())

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
    "Your job is to adjust rule PRESENCE — add missing rules and remove unsupported ones. "
    "Do NOT modify the content of any existing rule field.\n\n"
    "STEP 1 — Remove false positives: delete any rule whose condition, action, or measurement value "
    "is NOT grounded in explicit text from the document. "
    "Do NOT delete global or plant-wide rules merely because they lack a specific physical location. "
    "Do NOT delete rules simply because they resemble another rule; rules for the same sensor with "
    "different severity tiers or thresholds are intentional and must both be preserved.\n\n"
    "STEP 2 — Add false negatives: identify every explicit constraint, operational requirement, "
    "threshold definition, or maintenance trigger stated in the text that has no corresponding rule "
    "in the extracted list, and add it.\n\n"
    "Return valid JSON: {\"rules\": [ ... ]}. "
    "Copy all existing rules that pass STEP 1 exactly as received — do not rewrite any field."
)

# ── LANGGRAPH STATE ─────────────────────────────────────────────────────────────

def _merge_rules(left: list[dict], right: list[dict] | None) -> list[dict]:
    """Custom LangGraph reducer: None resets the accumulator, a list appends to it.

    The dual behaviour is intentional:
      - Append mode  : accumulates results across parallel fan-in branches.
      - Reset mode   : returning None from a node clears the list before a retry,
                       preventing stale results from a previous extraction pass
                       from polluting the next one.
    """
    if right is None:
        return []
    return (left or []) + right

class PipelineState(TypedDict, total=False):
    source_file:     str
    run_id:          str
    raw_text:        str
    doc_type:        str
    # Annotated with _merge_rules so LangGraph accumulates miner output across retries
    # and a None write cleanly resets the list without requiring an explicit clear node.
    extracted_rules: Annotated[list[dict], _merge_rules]
    merged_rules:    list[dict]
    graph_edges:     list[dict]   
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
    """Append judge feedback to the miner's user prompt on retry.

    Injecting specific gap descriptions (e.g. "missing threshold rules for ST03_TMP")
    directs the model toward the identified deficiencies rather than repeating the
    same blind extraction, which is unlikely to correct the same gaps.
    """
    feedback    = state.get("judge_feedback", {})
    suggestions = feedback.get("suggestions", "")
    gaps        = feedback.get("gaps", [])
    if not gaps and not suggestions:
        return ""
    return (
        f"\n\nATTENTION — previous extraction was flagged (gaps: {gaps}). "
        f"{suggestions} Be thorough and extract every rule present in the document."
    )

# ── STAGE 1: CLASSIFIER (kept only for judge checklists) ────────────────────────

def node_document_type_classifier(state: PipelineState) -> dict:
    """Classify document type via LLM. Does NOT affect miner selection."""
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
    raw, _ = llm_call(MODEL_NODE_CLASSIFIER, messages)
    try:
        doc_type = json.loads(raw).get("doc_type", "narrative").lower()
    except Exception:
        doc_type = "narrative"
    if doc_type not in ("tabular", "narrative", "matrix", "mixed"):
        doc_type = "narrative"
    print(f"      → {doc_type.upper()}")
    return {"doc_type": doc_type, "extracted_rules": None}  # reset extracted_rules

# ── STAGE 2: UNIVERSAL RULE MINER (single semantic extractor) ──────────────────

def node_universal_miner(state: PipelineState) -> dict:
    """Extract all rule types from the document, regardless of layout."""
    print("    [UniversalMiner] Extracting rules semantically...")
    messages = [
        {"role": "system", "content": UNIVERSAL_MINER_SYSTEM},
        {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT] + _retry_note(state)},
    ]
    raw, _ = llm_call(MODEL_NODE_MINER, messages)
    rules = parse_rules(raw)
    print(f"      → {len(rules)} rules")
    return {"extracted_rules": rules}

# ── STAGE 3: MERGER ─────────────────────────────────────────────────────────────

def node_consensus_merger(state: PipelineState) -> dict:
    """Deterministic-first merge strategy to minimise LLM calls.

    LLM adjudication is expensive and slow. Three deterministic branches handle
    the common cases — singletons, complementary threshold fields, and exact
    content duplicates — without any model call. Only genuine conflicts (same
    class/station/sensorType, different required action) are queued and resolved
    in a single batched LLM call at the end.
    """
    print("    [Merger] Merging...")
    all_rules = state.get("extracted_rules", [])

    if not all_rules:
        print("    [Merger] 0 rules — falling back to universal miner...")
        messages = [
            {"role": "system", "content": UNIVERSAL_MINER_SYSTEM},
            {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT]},
        ]
        raw, _ = llm_call(MODEL_NODE_MERGER, messages)
        fallback = parse_rules(raw)
        print(f"      → {len(fallback)} rules (fallback)")
        return {"merged_rules": fallback}

    # Step 1 — Group by (class, station, sensorType): rules that share all three
    # keys describe the same physical constraint and are candidates for merging.
    groups: dict[tuple, list[dict]] = {}
    for r in all_rules:
        key = (
            r.get("class", "").strip(),
            r.get("station", "").strip().upper(),
            r.get("sensorType", "").strip().upper(),
        )
        groups.setdefault(key, []).append(r)

    output: list[dict] = []
    conflict_groups: list[dict] = []

    # Step 2 — Process each group through a cascade of deterministic branches;
    # only escalate to LLM (Branch D) when no deterministic resolution is possible.
    for key, candidates in groups.items():
        cls = key[0]

        # Branch A — singleton: nothing to merge
        if len(candidates) == 1:
            output.append(candidates[0])
            continue

        # Branch B — ThresholdRule complementary merge (no LLM)
        # The miner often emits a WARNING-only rule and a CRITICAL-only rule for the
        # same sensor. When the two rules are complementary (one has crit fields, the
        # other has warn fields, and neither is already complete), merge them
        # deterministically into a single four-field rule to avoid double-counting.
        if (
            cls == "ThresholdRule"
            and len(candidates) == 2
            and not _has_all_four(candidates[0])
            and not _has_all_four(candidates[1])
            and (
                (_has_crit_fields(candidates[0]) and _has_warn_fields(candidates[1]))
                or (_has_warn_fields(candidates[0]) and _has_crit_fields(candidates[1]))
            )
        ):
            crit_src = candidates[0] if _has_crit_fields(candidates[0]) else candidates[1]
            warn_src = candidates[1] if _has_crit_fields(candidates[0]) else candidates[0]
            merged = dict(crit_src)
            for field in ("warnHi", "warnLo"):
                if not merged.get(field) and warn_src.get(field):
                    merged[field] = warn_src[field]
            for field in ("station", "sensor", "sensorType", "condition", "action", "unit"):
                if not merged.get(field) and warn_src.get(field):
                    merged[field] = warn_src[field]
            merged["severity"] = "CRITICAL"
            output.append(merged)
            continue

        # Branch C — exact content duplicate detection (no LLM)
        # Content signature across six fields catches re-extractions of the same rule
        # under slightly different wording; keep the candidate with the most populated fields.
        seen_sigs: dict[str, dict] = {}
        for r in candidates:
            sig = "|".join([
                r.get("class", "").lower(),
                r.get("station", "").lower(),
                r.get("sensorType", "").lower(),
                r.get("severity", "").lower(),
                _norm_text(r.get("condition", "")),
                _norm_text(r.get("action", "")),
            ])
            if sig not in seen_sigs or _n_nonempty(r) > _n_nonempty(seen_sigs[sig]):
                seen_sigs[sig] = r

        deduped = list(seen_sigs.values())
        if len(deduped) == 1:
            output.append(deduped[0])
            continue

        # Branch D — genuine conflict: same key but different actions → queue for LLM
        conflict_groups.append({
            "group_key": {"class": key[0], "station": key[1], "sensorType": key[2]},
            "candidates": deduped,
        })

    # Step 3 — one batched LLM call for all conflict groups: cheaper than one call per conflict
    if conflict_groups:
        print(f"      [Merger] {len(conflict_groups)} conflict group(s) → LLM adjudication")
        json_conflicts = json.dumps(conflict_groups, indent=2)[:12000]
        messages = [
            {"role": "system", "content": CONFLICT_ADJUDICATOR_SYSTEM},
            {"role": "user",   "content": json_conflicts},
        ]
        raw, _ = llm_call(MODEL_NODE_MERGER, messages)
        try:
            resolved = json.loads(
                re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            ).get("resolved", [])
        except Exception:
            resolved = []

        # Step 4 — post-LLM categorical guard: LLMs occasionally corrupt categorical fields
        # (class, severity, station) during merging; restore them from the richest original candidate.
        all_originals = [r for grp in conflict_groups for r in grp["candidates"]]
        if resolved:
            for r in resolved:
                if isinstance(r, dict):
                    output.append(_guard_fields(r, all_originals))
        else:
            # LLM failed: keep all conflict candidates unchanged
            for grp in conflict_groups:
                output.extend(grp["candidates"])

    print(f"      → {len(output)} rules after merge")
    return {"merged_rules": output}

# ── STAGE 4: JUDGE ──────────────────────────────────────────────────────────────

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
    raw, _ = llm_call(MODEL_NODE_EVALUATOR, messages)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        feedback = json.loads(cleaned)
    except Exception:
        feedback = {"pass": True, "gaps": [], "suggestions": ""}

    print(f"      → Pass: {feedback.get('pass', True)}, Gaps: {feedback.get('gaps', [])}")
    return {"judge_feedback": feedback}

# ── STAGE 5: NORMALIZER ─────────────────────────────────────────────────────────

def node_identifier_canonicalizer(state: PipelineState) -> dict:
    """Assign canonical ruleIds while preserving all other fields.

    Only the ruleId field is sourced from the LLM output; all semantic fields
    (class, station, condition, action, …) come from the original merged_rules.
    This limits the blast radius of an LLM normalization error to identifiers only.
    Position-aligned assignment requires the model to return exactly len(merged_rules)
    entries; any count mismatch falls back to the originals unchanged.
    """
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
    raw, _ = llm_call(MODEL_NODE_CANONICALIZER, messages)
    normalized = parse_rules(raw)

    # Extract only ruleIds from LLM output; all other fields come from originals.
    # Position-aligned when count matches; fall back to originals on mismatch.
    if normalized and len(normalized) == len(merged):
        final = [
            {**orig, "ruleId": norm.get("ruleId", orig.get("ruleId", ""))}
            for orig, norm in zip(merged, normalized)
        ]
    else:
        final = merged

    print(f"      → {len(final)} rules normalized")
    return {"final_rules": final}

# ── STAGE 6: CONTENT REFINER (FP/FN scrubber) ───────────────────────────────────

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
    raw, _ = llm_call(MODEL_NODE_VALIDATOR, messages)
    validated = parse_rules(raw)
    llm_result = validated if validated else rules

    llm_removed = max(0, len(rules) - len(llm_result))
    llm_added   = max(0, len(llm_result) - len(rules))

    # Deterministic post-filter: class validation, numeric completeness, dedup
    result = _deterministic_post_filter(llm_result)
    det_removed = len(llm_result) - len(result)

    print(f"      → {len(result)} rules after validation "
          f"(LLM: -{llm_removed} FPs +{llm_added} FNs | det: -{det_removed})")
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
    """Deterministic router that separates two semantically distinct knowledge types.

    AuthorizationRule encodes binary role×zone permissions extracted from access-control
    grids. These are Neo4j graph edges (authorized_for relationships), not independently
    actionable operational rules. Routing them away from the rule corpus before scoring
    prevents inflated rule counts and distorted F1 evaluation. All other classes —
    including any non-standard classes the LLM may emit — flow to QualityGapEvaluator
    so the judge and normalizer can handle them without hardcoded class assumptions.
    """
    merged = state.get("merged_rules", [])
    rule_nodes: list[dict] = []
    graph_edges: list[dict] = []

    for r in merged:
        if r.get("class") == "AuthorizationRule":
            graph_edges.append({**r, "_edge_type": "authorized_for"})
        else:
            rule_nodes.append(r)

    print(
        f"    [KnowledgeRouter] {len(rule_nodes)} rule_nodes → QualityGapEvaluator | "
        f"{len(graph_edges)} graph_edges → Neo4j (authorized_for)"
    )
    return {"merged_rules": rule_nodes, "graph_edges": graph_edges}

# ── CONDITIONAL EDGES ───────────────────────────────────────────────────────────

def route_after_judge(state: PipelineState) -> str:
    if state.get("judge_feedback", {}).get("pass", True):
        return "IdentifierCanonicalizer"
    if state.get("retry_count", 0) < MAX_JUDGE_RETRIES:
        return "ResetAndRetry"
    print("    [Judge] Max retries reached — proceeding to normalization.")
    return "IdentifierCanonicalizer"

# ── BUILD GRAPH ─────────────────────────────────────────────────────────────────

def build_pipeline():
    """Full pipeline with universal miner and no document‑type routing."""
    graph = StateGraph(PipelineState)

    graph.add_node("DocumentTypeClassifier",    node_document_type_classifier)
    graph.add_node("UniversalRuleMiner",        node_universal_miner)
    graph.add_node("ConsensusMerger",           node_consensus_merger)
    graph.add_node("KnowledgeRouter",           node_knowledge_router)
    graph.add_node("Validator",                 node_validate)
    graph.add_node("QualityGapEvaluator",       node_quality_gap_evaluator)
    graph.add_node("IdentifierCanonicalizer",   node_identifier_canonicalizer)
    graph.add_node("ResetAndRetry",             node_reset_retry)

    graph.add_edge(START, "DocumentTypeClassifier")
    graph.add_edge("DocumentTypeClassifier", "UniversalRuleMiner")
    graph.add_edge("UniversalRuleMiner",     "ConsensusMerger")
    graph.add_edge("ConsensusMerger",        "KnowledgeRouter")
    graph.add_edge("KnowledgeRouter",        "Validator")
    graph.add_edge("Validator",              "QualityGapEvaluator")

    graph.add_conditional_edges(
        "QualityGapEvaluator", route_after_judge,
        ["IdentifierCanonicalizer", "ResetAndRetry"],
    )
    graph.add_edge("ResetAndRetry",           "DocumentTypeClassifier")  # re-run from classifier
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
        print("    [SimplePassthrough] 0 rules — fallback to universal miner...")
        messages = [
            {"role": "system", "content": UNIVERSAL_MINER_SYSTEM},
            {"role": "user",   "content": state["raw_text"][:SOP_TEXT_LIMIT]},
        ]
        raw, _ = llm_call(MODEL_NODE_PASSTHROUGH, messages)
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

    Each variant is constructed as an independent LangGraph graph rather than
    parameterising the full pipeline at runtime because LangGraph compiles the
    node registry and edge table at build time — there is no mechanism to toggle
    nodes on/off after compilation without rebuilding the graph.

    KnowledgeRouter is excluded from the ablation set because it performs a
    correctness transformation (separating graph edges from rule nodes) rather
    than an optional quality-improvement step; removing it would corrupt scoring.

    ablation must be one of ABLATION_OPTIONS:
      no_consensus_merger         — replace ConsensusMerger with deterministic dedup only
      no_quality_gap_evaluator    — remove the judge loop and all retries
      no_identifier_canonicalizer — copy merged_rules → final_rules without LLM normalisation
      no_validator                — connect KnowledgeRouter directly to the Judge
    """
    if ablation not in ABLATION_OPTIONS:
        raise ValueError(f"Unknown ablation {ablation!r}. Choose from: {ABLATION_OPTIONS}")

    graph = StateGraph(PipelineState)

    if ablation == "no_consensus_merger":
        graph.add_node("DocumentTypeClassifier",  node_document_type_classifier)
        graph.add_node("UniversalRuleMiner",      node_universal_miner)
        graph.add_node("SimplePassthroughMerge",  node_simple_passthrough_merge)
        graph.add_node("KnowledgeRouter",         node_knowledge_router)
        graph.add_node("Validator",               node_validate)
        graph.add_node("QualityGapEvaluator",     node_quality_gap_evaluator)
        graph.add_node("IdentifierCanonicalizer", node_identifier_canonicalizer)
        graph.add_node("ResetAndRetry",           node_reset_retry)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_edge("DocumentTypeClassifier", "UniversalRuleMiner")
        graph.add_edge("UniversalRuleMiner",     "SimplePassthroughMerge")
        graph.add_edge("SimplePassthroughMerge", "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter",        "Validator")
        graph.add_edge("Validator",              "QualityGapEvaluator")
        graph.add_conditional_edges("QualityGapEvaluator", route_after_judge,
            ["IdentifierCanonicalizer", "ResetAndRetry"])
        graph.add_edge("ResetAndRetry",           "DocumentTypeClassifier")
        graph.add_edge("IdentifierCanonicalizer", END)

    elif ablation == "no_quality_gap_evaluator":
        graph.add_node("DocumentTypeClassifier",  node_document_type_classifier)
        graph.add_node("UniversalRuleMiner",      node_universal_miner)
        graph.add_node("ConsensusMerger",         node_consensus_merger)
        graph.add_node("KnowledgeRouter",         node_knowledge_router)
        graph.add_node("Validator",               node_validate)
        graph.add_node("IdentifierCanonicalizer", node_identifier_canonicalizer)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_edge("DocumentTypeClassifier", "UniversalRuleMiner")
        graph.add_edge("UniversalRuleMiner",     "ConsensusMerger")
        graph.add_edge("ConsensusMerger",        "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter",        "Validator")
        graph.add_edge("Validator",              "IdentifierCanonicalizer")
        graph.add_edge("IdentifierCanonicalizer", END)

    elif ablation == "no_identifier_canonicalizer":
        graph.add_node("DocumentTypeClassifier", node_document_type_classifier)
        graph.add_node("UniversalRuleMiner",     node_universal_miner)
        graph.add_node("ConsensusMerger",        node_consensus_merger)
        graph.add_node("KnowledgeRouter",        node_knowledge_router)
        graph.add_node("Validator",              node_validate)
        graph.add_node("QualityGapEvaluator",    node_quality_gap_evaluator)
        graph.add_node("PassthroughNormalize",   node_passthrough_normalize)
        graph.add_node("ResetAndRetry",          node_reset_retry)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_edge("DocumentTypeClassifier", "UniversalRuleMiner")
        graph.add_edge("UniversalRuleMiner",     "ConsensusMerger")
        graph.add_edge("ConsensusMerger",        "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter",        "Validator")
        graph.add_edge("Validator",              "QualityGapEvaluator")
        graph.add_conditional_edges("QualityGapEvaluator", _route_no_canonicalizer,
            ["PassthroughNormalize", "ResetAndRetry"])
        graph.add_edge("ResetAndRetry",        "DocumentTypeClassifier")
        graph.add_edge("PassthroughNormalize", END)

    elif ablation == "no_validator":
        graph.add_node("DocumentTypeClassifier",  node_document_type_classifier)
        graph.add_node("UniversalRuleMiner",      node_universal_miner)
        graph.add_node("ConsensusMerger",         node_consensus_merger)
        graph.add_node("KnowledgeRouter",         node_knowledge_router)
        graph.add_node("QualityGapEvaluator",     node_quality_gap_evaluator)
        graph.add_node("IdentifierCanonicalizer", node_identifier_canonicalizer)
        graph.add_node("ResetAndRetry",           node_reset_retry)

        graph.add_edge(START, "DocumentTypeClassifier")
        graph.add_edge("DocumentTypeClassifier", "UniversalRuleMiner")
        graph.add_edge("UniversalRuleMiner",     "ConsensusMerger")
        graph.add_edge("ConsensusMerger",        "KnowledgeRouter")
        graph.add_edge("KnowledgeRouter",        "QualityGapEvaluator")
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
    print(f"  node_classifier    = {MODEL_NODE_CLASSIFIER}")
    print(f"  node_miner         = {MODEL_NODE_MINER}")
    print(f"  node_merger        = {MODEL_NODE_MERGER}")
    print(f"  node_evaluator     = {MODEL_NODE_EVALUATOR}")
    print(f"  node_canonicalizer = {MODEL_NODE_CANONICALIZER}")
    print(f"  node_validator     = {MODEL_NODE_VALIDATOR}")
    print(f"  node_passthrough   = {MODEL_NODE_PASSTHROUGH}")
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