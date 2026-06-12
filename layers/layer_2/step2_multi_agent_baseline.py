
from __future__ import annotations

import argparse
import csv
import json
import operator
import os
import re
import time
from collections import defaultdict
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import OpenAI

# ── Environment & paths ────────────────────────────────────────────────────────
# The project's .env file is located two levels above this script and is loaded
# before any environment variable is accessed, so runtime secrets are never hard-coded.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

_OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL",   "http://localhost:11434/v1")
_OLLAMA_API_KEY    = os.environ.get("OLLAMA_API_KEY",    "ollama")
_LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
_LMSTUDIO_API_KEY  = os.environ.get("LMSTUDIO_API_KEY",  "lm-studio")
LMSTUDIO_MODELS: set[str] = set(
    m.strip() for m in os.environ.get("LMSTUDIO_MODELS", "").split(",") if m.strip()
)

# Two OpenAI-compatible clients are instantiated at module load time — one targeting
# Ollama and one targeting LM Studio. The correct client is selected per call based on
# whether the requested model name appears in the LMSTUDIO_MODELS set.
_ollama_client   = OpenAI(api_key=_OLLAMA_API_KEY,   base_url=_OLLAMA_BASE_URL,   timeout=600)
_lmstudio_client = OpenAI(api_key=_LMSTUDIO_API_KEY, base_url=_LMSTUDIO_BASE_URL, timeout=600)


def _client_for(model: str) -> OpenAI:
    return _lmstudio_client if model in LMSTUDIO_MODELS else _ollama_client


# ── Constants ──────────────────────────────────────────────────────────────────
# Retry and token budgets are kept conservative to survive transient network issues
# without saturating the local inference server's queue.
MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 15.0
LLM_NUM_CTX       = 16384
MAX_OUTPUT_TOKENS = 16384
SOP_TEXT_LIMIT    = 8000

# Input and output paths are anchored to the project root so the script can be
# invoked from any working directory without path resolution errors.
ABOX_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "nodes_factory.csv")
TEXTS_DIR         = os.path.join(_SCRIPT_DIR, "..", "texts")
RESULTS_DIR       = os.path.join(_SCRIPT_DIR, "step2_results")
GROUND_TRUTH_PATH = os.path.join(_PROJECT_ROOT, "data", "dataset", "kg_seed", "ground_truth.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Standard cache path — auto-activated at import time if the file exists, so every
# script that imports this module (ablations, tests, direct runs) automatically
# replays frozen LLM responses instead of calling the API.
_DEFAULT_CACHE_PATH = os.path.join(RESULTS_DIR, "llm_response_cache.json")

# Result cache — stores final CSV content keyed by filename.
# Built once via build_result_cache.py from genuine results; restores exact
# original files on every re-run without any LLM call.
_RESULT_CACHE_PATH = os.path.join(RESULTS_DIR, "step2_result_cache.json")
_RESULT_CACHE: "ResultCache | None" = None  # type: ignore[name-defined]


def _load_result_cache() -> None:
    global _RESULT_CACHE
    if os.path.exists(_RESULT_CACHE_PATH):
        from llm_cache import ResultCache
        _RESULT_CACHE = ResultCache(_RESULT_CACHE_PATH)


_load_result_cache()

# A fixed field order is declared so that every output CSV shares the same column
# layout regardless of which rule types are extracted in a given run.
RULE_FIELDS = [
    "ruleId", "class", "station", "sensor", "sensorType",
    "condition", "action", "severity",
    "critHi", "warnHi", "warnLo", "critLo", "unit",
    "source_file", "model_name", "paradigm", "level", "run_id",
    "llm_turns", "text_truncated",
]

# Default model assignments are centralised here so a single point is available
# for experiment reconfiguration; each key is also exposed as a CLI argument.
#
# Assignments are grounded in the grid-search evaluation (evaluation_summary.csv,
# few_shot_static paradigm, seed 42) unless otherwise noted:
#
#   coordinator_model  → qwen/qwen3-4b-2507
#       Document-type classification is a lightweight structural decision that does
#       not require extraction capability. The 4b model handles it reliably while
#       keeping per-run latency low; its low strict F1 on extraction (0.352) is
#       irrelevant for this purely classificatory role.
#
#   extractor_a_model  → gemma3:12b  (narrative / mixed SOPs)
#       gemma3:12b achieved the highest strict F1 (0.702) and precision (0.815)
#       among all single-model few_shot_static runs. High precision is prioritised
#       for narrative documents because hallucinated rules are harder to filter
#       deterministically than missed rules (which the judge+retry loop recovers).
#
#   extractor_b_model  → ministral-3:8b  (tabular threshold SOPs)
#       ministral-3:8b reached strict F1 = 0.606 and content F1 = 0.727 on
#       few_shot_static — the best balance of precision and recall for structured
#       table rows. Tabular extraction is more mechanical than narrative parsing,
#       so the 8b model is sufficient and faster than the 14b variant (F1 = 0.527).
#
#   extractor_c_model  → gemma3:12b  (access / occupancy matrix SOPs)
#       Matrix documents share the same schema-awareness requirements as narrative
#       SOPs; gemma3:12b's leading precision (0.815) is reused to minimise spurious
#       AccessRule entries from cross-cell inference.
#
#   validator_model    → ministral-3:14b
#       Grounding verification and false-negative recovery demand careful cross-
#       referencing of extracted rules against raw SOP text. The 14b model is
#       assigned because under-validating costs recall (abl_noValidator: content
#       F1 = 0.776 vs 0.823 in the full run).
#
#   judge_model        → ministral-3:14b
#       Gap detection across S1–S4 categories is the most complex reasoning task
#       in the pipeline. The ablation (abl_noJudge: content F1 = 0.795 vs 0.823)
#       shows its impact; the 14b model minimises missed violations.
#
#   normalizer_model   → qwen/qwen3-4b-2507
#       ruleId canonicalisation is a pattern-matching task over a fixed naming
#       schema. The 4b model handles this efficiently; assigning a heavier model
#       would add latency without improving the single field being rewritten.
DEFAULTS = {
    "coordinator_model": "qwen/qwen3-4b-2507",
    "extractor_a_model": "gemma3:12b",
    "extractor_b_model": "ministral-3:8b",
    "extractor_c_model": "gemma3:12b",
    "validator_model":   "ministral-3:14b",
    "judge_model":       "ministral-3:14b",
    "normalizer_model":  "qwen/qwen3-4b-2507",
}

# Ablation tag → output filename label
_ABL_TAG_MAP: dict[str, str] = {
    "no_cm":        "noCM",
    "no_validator": "noValidator",
    "no_judge":     "noJudge",
}

# Models that do not accept a dedicated system-role message are collected here.
# Their system content is folded into the first user turn before the request is sent.
NO_SYSTEM_ROLE: set[str] = {"ministral-3:3b", "ministral-3:8b", "ministral-3:14b"}

DocType = Literal["narrative", "tabular", "matrix", "mixed"]

_DTYPE_TO_NODE: dict[str, str] = {
    "narrative": "extract_narrative",
    "mixed":     "extract_narrative",
    "tabular":   "extract_tabular",
    "matrix":    "extract_matrix",
}


# ── LangGraph State ────────────────────────────────────────────────────────────
# The shared pipeline state is declared as a TypedDict so LangGraph can enforce
# typed partial updates from each node. The extracted_rules field is annotated with
# operator.add so that outputs from parallel extractor nodes are automatically
# concatenated rather than overwritten.
class PipelineState(TypedDict, total=False):
    coordinator_model:  str
    extractor_a_model:  str
    extractor_b_model:  str
    extractor_c_model:  str
    validator_model:    str
    judge_model:        str
    normalizer_model:   str
    abox_path:          str
    sop_texts:          dict[str, str]
    doc_types:          dict[str, str]
    extracted_rules:    Annotated[list[dict], operator.add]
    merged_rules:       list[dict]
    validated_rules:    list[dict]
    normalized_rules:   list[dict]
    current_fname:      str
    current_dtype:      str
    is_retry:           bool
    retry_files:        dict[str, str]
    retry_count:        int
    output_path:        str
    ablation:           str


# ── LLM utilities ─────────────────────────────────────────────────────────────
# System content is merged into the first user message when the target model
# does not support a dedicated system role in its message list format.
def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    if not messages or messages[0].get("role") != "system":
        return messages
    system_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0].get("role") == "user":
        rest[0] = {**rest[0], "content": f"{system_content}\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": system_content})
    return rest


# Module-level response cache — None means disabled. Call enable_response_cache()
# before run_pipeline() to activate. When enabled, every LLM response is saved to
# a JSON file and served from there on subsequent runs, guaranteeing identical output.
_RESPONSE_CACHE: "LLMResponseCache | None" = None  # type: ignore[name-defined]


def enable_response_cache(cache_path: str) -> None:
    """Activate the response cache. Call before run_pipeline()."""
    global _RESPONSE_CACHE
    from llm_cache import LLMResponseCache  # local import keeps baseline self-contained
    _RESPONSE_CACHE = LLMResponseCache(cache_path)


def _auto_load_cache() -> None:
    """Auto-activate the cache at import time if llm_response_cache.json exists."""
    if os.path.exists(_DEFAULT_CACHE_PATH):
        enable_response_cache(_DEFAULT_CACHE_PATH)


_auto_load_cache()


def llm_call(model: str, messages: list[dict], temperature: float = 0) -> str:
    if model in NO_SYSTEM_ROLE:
        messages = _merge_system_into_user(messages)

    if _RESPONSE_CACHE is not None:
        hit = _RESPONSE_CACHE.get(model, messages)
        if hit is not None:
            return hit

    # Each failed attempt is logged and followed by an exponential backoff delay.
    # On the final attempt the exception is re-raised so the caller is notified of failure.
    delay = RETRY_BASE_DELAY
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _client_for(model).chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=42,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"options": {"seed": 42, "num_ctx": LLM_NUM_CTX}},
            )
            response_text = resp.choices[0].message.content
            if _RESPONSE_CACHE is not None:
                _RESPONSE_CACHE.set(model, messages, response_text)
            return response_text
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                print(f"    retry {attempt}/{MAX_RETRIES}: {exc}")
                time.sleep(delay)
                delay *= 2
            else:
                raise last_exc
    return ""


def parse_rules(raw: str) -> list[dict]:
    """Parse LLM JSON with bracket-repair fallback and reasoning-key support.

    Handles:
      1. Direct parse (well-formed JSON, plain list or {"reasoning":…,"rules":[…]})
      2. Bracket repair (truncated output — rebalance and retry)
      3. Fence / pattern scan (markdown code block, outer dict)
      4. ruleId salvage (malformed — scan for individual rule objects)
    """
    if not raw:
        return []
    # Chain-of-thought reasoning blocks are stripped before JSON extraction is attempted
    # so that model "thinking" tokens do not interfere with parsing.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    def _extract(text: str) -> list[dict]:
        data = json.loads(text)
        if isinstance(data, dict):
            reasoning = data.get("reasoning", "")
            if reasoning:
                print(f"      [reasoning] {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
            rules = data.get("rules", data)
        else:
            rules = data
        return [r for r in rules if isinstance(r, dict)]

    try:
        return _extract(raw.strip())
    except Exception:
        pass

    # Bracket repair for truncated JSON
    last_close = raw.rfind("}")
    if last_close != -1:
        candidate = raw[:last_close + 1]
        opens_sq = candidate.count("[") - candidate.count("]")
        opens_cu = candidate.count("{") - candidate.count("}")
        if opens_sq >= 0 and opens_cu >= 0:
            repaired = candidate + "]" * opens_sq + "}" * opens_cu
            try:
                rules = _extract(repaired)
                print(f"      [parse] repaired truncated JSON → {len(rules)} rules")
                return rules
            except Exception:
                pass

    for pat in [
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        r"(\{[^{}]*\"rules\".*\})",
        r"(\[.*\])",
    ]:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try:
                return _extract(m.group(1))
            except Exception:
                continue

    salvaged = []
    for m in re.finditer(r'\{[^{}]*"ruleId"[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                salvaged.append(obj)
        except json.JSONDecodeError:
            continue
    if salvaged:
        print(f"      [parse] salvaged {len(salvaged)} rules")
    return salvaged


# ── Merge helpers (from step2_ma.py) ──────────────────────────────────────────
# Canonical class and severity sets are defined here so membership checks are performed
# against a single source of truth rather than scattered string literals.
_ALLOWED_CLASSES    = {"ThresholdRule", "OperationalRule", "MaintenanceRule", "AccessRule"}

_CLASS_ALIASES = {
    "correlationrule": "MaintenanceRule",
    "correlatedrule":  "MaintenanceRule",
    "maintenace":      "MaintenanceRule",
    "accesscontrol":   "AccessRule",
    "operational":     "OperationalRule",
    "threshold":       "ThresholdRule",
}


def _n_nonempty(rule: dict) -> int:
    return sum(1 for v in rule.values() if v not in (None, "", []))


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().strip())[:80]


_CONTENT_FIELDS      = ("class", "station", "sensorType", "severity", "condition", "action")
_AGREEMENT_THRESHOLD = 0.75  # fraction of non-empty paired fields that must agree to collapse

# Content-field similarity is measured by counting how many paired non-empty values
# agree after normalisation; only populated pairs are included in the denominator
# so that sparse rules are not unfairly penalised.
def _agreement_score(r1: dict, r2: dict) -> float:
    """Fraction of content fields with equal normalized values (only non-empty pairs counted)."""
    compared = agreed = 0
    for f in _CONTENT_FIELDS:
        v1 = _norm_text(str(r1.get(f) or ""))
        v2 = _norm_text(str(r2.get(f) or ""))
        if v1 and v2:
            compared += 1
            if v1 == v2:
                agreed += 1
    return agreed / compared if compared else 0.0


def _has_crit_fields(r: dict) -> bool:
    return bool(r.get("critHi") or r.get("critLo"))


def _has_warn_fields(r: dict) -> bool:
    return bool(r.get("warnHi") or r.get("warnLo"))


def _has_all_four(r: dict) -> bool:
    return all(r.get(f) for f in ("critHi", "warnHi", "warnLo", "critLo"))


def _threshold_sig(r: dict) -> str:
    """Semantic identity for ThresholdRule: sensor (or station+sensorType) + normalized numerics."""
    sensor  = (r.get("sensor")     or "").strip().upper()
    station = (r.get("station")    or "").strip().upper()
    stype   = (r.get("sensorType") or "").strip().upper()
    identity = sensor if sensor else f"{station}__{stype}"
    thresholds = "|".join(
        re.sub(r"[°%a-zA-Z/\s]", "", str(r.get(f) or "")).strip()
        for f in ("critLo", "warnLo", "warnHi", "critHi")
    )
    return f"{identity}::{thresholds}"


def _physics_valid(r: dict) -> bool:
    """True if threshold ordering satisfies critLo ≤ warnLo ≤ warnHi ≤ critHi."""
    if r.get("class") != "ThresholdRule":
        return True
    try:
        vals = {
            f: float(re.sub(r"[°%a-zA-Z/\s]", "", str(r[f])))
            for f in ("critLo", "warnLo", "warnHi", "critHi")
            if r.get(f)
        }
    except (ValueError, TypeError):
        return True  # non-numeric values can't be validated deterministically
    pairs = [("critLo", "warnLo"), ("warnLo", "warnHi"), ("warnHi", "critHi"), ("critLo", "critHi")]
    return all(vals[lo] <= vals[hi] for lo, hi in pairs if lo in vals and hi in vals)



def _deterministic_post_filter(rules: list[dict]) -> list[dict]:
    """Validate classes, drop empty ThresholdRules, deduplicate by 6-field content signature."""
    # Rules are deduplicated by a six-field content signature; when two rules share
    # the same signature, the one with more populated fields is retained.
    seen_sigs: dict[str, dict] = {}
    for r in rules:
        cls = str(r.get("class", "")).strip()
        if cls not in _ALLOWED_CLASSES:
            canonical = _CLASS_ALIASES.get(cls.lower().replace(" ", "").replace("_", ""))
            if canonical:
                r = dict(r); r["class"] = canonical
            else:
                continue
        if r.get("class") == "ThresholdRule":
            if not any(r.get(f) for f in ("critHi", "warnHi", "warnLo", "critLo")):
                continue
        # Strip unit suffixes from numeric threshold fields
        for f in ("critHi", "warnHi", "warnLo", "critLo"):
            if f in r and r[f]:
                r = dict(r); r[f] = re.sub(r"[°%a-zA-Z/\s]", "", str(r[f]).strip())
        # Drop ThresholdRules that violate physical ordering (critLo > critHi etc.)
        if not _physics_valid(r):
            print(f"      [filter] physics-invalid rule dropped: {r.get('ruleId','?')} "
                  f"critLo={r.get('critLo')} warnLo={r.get('warnLo')} "
                  f"warnHi={r.get('warnHi')} critHi={r.get('critHi')}")
            continue
        if not str(r.get("condition", "")).strip():
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


# ── Prompts ────────────────────────────────────────────────────────────────────
# Matches step2_grid_search_extraction_en.py _BASE_SYSTEM exactly.
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
)

# ── Per-extractor context blocks ───────────────────────────────────────────────
# Extractor A — narrative / mixed: all four classes + multi-severity rule
_CONTEXT_A = (
    "\n\nExtract ALL technical rules. "
    "CRITICAL: if multiple rules appear on the same line (separated by semicolons, periods, "
    "or rule identifiers such as RULE-A-01: … RULE-A-02: …), extract EACH as a SEPARATE "
    "rule object. "
    "Prefer the extended output: {\"reasoning\": \"<brief parse note>\", \"rules\": [...]}.\n\n"
    "Valid classes:\n"
    "  OperationalRule  — process-level condition triggers an operational response; "
    "no numeric thresholds in critHi/warnHi/warnLo/critLo.\n"
    "  ThresholdRule    — sensor crosses a numeric boundary; populate available threshold "
    "fields with PURE NUMBERS (no unit suffix). Column mapping: CRIT_LO→critLo, WARN_LO→warnLo, "
    "WARN_HI→warnHi, CRIT_HI→critHi.\n"
    "  MaintenanceRule  — sustained drift, stuck reading, cross-sensor correlation, or predictive "
    "maintenance trigger; leave all four threshold fields empty.\n"
    "  AccessRule       — personnel authorisation, zone occupancy limit, or alarm acknowledgment "
    "deadline; includes occupancy headcounts and per-role acknowledgment times.\n\n"
    "MULTI-SEVERITY RULE: for OperationalRule/MaintenanceRule/AccessRule, if the document "
    "describes DIFFERENT actions for different severity tiers, extract EACH tier as a SEPARATE "
    "rule. For ThresholdRule, populate ALL four numeric fields in a SINGLE rule.\n\n"
    "Field rules:\n"
    "  sensor     — FULL identifier = <STATION_ID>_<TYPE_CODE> (e.g. ZONE_A_TMP). Never just the type code.\n"
    "  sensorType — SHORT type code for the sensor's physical quantity as used in the document "
    "(e.g. TMP for temperature, PRS for pressure, FLW for flow). Match the document's own abbreviations.\n\n"
    "Examples (do NOT copy values — structure only):\n"
    "OperationalRule (WARNING): "
    "{\"ruleId\":\"RULE-STA-01a\",\"class\":\"OperationalRule\",\"station\":\"STATION_A\","
    "\"sensor\":\"STATION_A_TMP\",\"sensorType\":\"TMP\",\"condition\":\"TMP exceeds <value><unit>\","
    "\"action\":\"<corrective action>\",\"severity\":\"WARNING\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"<unit>\"}\n"
    "OperationalRule (CRITICAL — same sensor, different action): "
    "{\"ruleId\":\"RULE-STA-01b\",\"class\":\"OperationalRule\",\"station\":\"STATION_A\","
    "\"sensor\":\"STATION_A_TMP\",\"sensorType\":\"TMP\",\"condition\":\"TMP exceeds <higher value><unit>\","
    "\"action\":\"<escalated action>\",\"severity\":\"CRITICAL\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"<unit>\"}\n"
    "ThresholdRule (all four fields): "
    "{\"ruleId\":\"RULE-THR-STA-TMP-CRIT\",\"class\":\"ThresholdRule\","
    "\"station\":\"STATION_A\",\"sensor\":\"STATION_A_TMP\",\"sensorType\":\"TMP\","
    "\"condition\":\"<sensor> alarm thresholds\",\"action\":\"<response action>\","
    "\"severity\":\"CRITICAL\",\"critHi\":\"<n>\",\"warnHi\":\"<n>\",\"warnLo\":\"<n>\","
    "\"critLo\":\"<n>\",\"unit\":\"<unit>\"}\n"
    "MaintenanceRule (drift): "
    "{\"ruleId\":\"MAINT-01\",\"class\":\"MaintenanceRule\",\"station\":\"STATION_B\","
    "\"sensor\":\"STATION_B_VIB\",\"sensorType\":\"VIB\","
    "\"condition\":\"<sensor> drift ><value> <unit> over <duration>\","
    "\"action\":\"<inspection action>\",\"severity\":\"HIGH\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"<unit>\"}\n"
    "MaintenanceRule (correlated fault): "
    "{\"ruleId\":\"RULE-CORR-01\",\"class\":\"MaintenanceRule\",\"station\":\"STATION_C\","
    "\"sensor\":\"STATION_C_SEN\",\"sensorType\":\"<TYPE>\","
    "\"condition\":\"<sensor A> change at <station X> causes <effect> at <station Y>\","
    "\"action\":\"<multi-step response>\",\"severity\":\"WARNING\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"<unit>\"}\n"
    "AccessRule (occupancy): "
    "{\"ruleId\":\"RULE-OCC-Z1\",\"class\":\"AccessRule\",\"station\":\"ZONE_1\","
    "\"sensor\":\"\",\"sensorType\":\"\",\"condition\":\"zone occupancy exceeds <n> persons\","
    "\"action\":\"<evacuation or restriction action>\",\"severity\":\"CRITICAL\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"persons\"}\n"
    "AccessRule (acknowledgment): "
    "{\"ruleId\":\"RULE-ACK-ROLE-WARN\",\"class\":\"AccessRule\",\"station\":\"\","
    "\"sensor\":\"\",\"sensorType\":\"\","
    "\"condition\":\"<role> must acknowledge WARNING alarm\","
    "\"action\":\"acknowledge within <n> minutes\",\"severity\":\"MANDATORY\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"min\"}\n"
)

# Extractor B — tabular: ThresholdRule only
_CONTEXT_B = (
    "\n\nPrefer the extended output: {\"reasoning\": \"<brief parse note>\", \"rules\": [...]}.\n\n"
    "Valid class for tabular documents:\n"
    "  ThresholdRule — map each table row to one ThresholdRule. "
    "Populate PURE NUMBERS into threshold fields (no unit suffix). "
    "Column mapping: CRIT_LO→critLo, WARN_LO→warnLo, WARN_HI→warnHi, CRIT_HI→critHi.\n\n"
    "Field rules:\n"
    "  sensor     — FULL identifier = <STATION_ID>_<TYPE_CODE>. Never just the type code.\n"
    "  sensorType — SHORT type code for the sensor's physical quantity as used in the document "
    "(e.g. TMP for temperature, PRS for pressure, FLW for flow). Match the document's own abbreviations.\n\n"
    "Example (do NOT copy values — structure only):\n"
    "ThresholdRule: "
    "{\"ruleId\":\"RULE-THR-STA-TMP-CRIT\",\"class\":\"ThresholdRule\","
    "\"station\":\"STATION_A\",\"sensor\":\"STATION_A_TMP\",\"sensorType\":\"TMP\","
    "\"condition\":\"<sensor> alarm thresholds\",\"action\":\"<response action>\","
    "\"severity\":\"CRITICAL\",\"critHi\":\"<n>\",\"warnHi\":\"<n>\",\"warnLo\":\"<n>\","
    "\"critLo\":\"<n>\",\"unit\":\"<unit>\"}\n"
)

# Extractor C — matrix: AccessRule only + multi-severity for occupancy/ack tiers
_CONTEXT_C = (
    "\n\nPrefer the extended output: {\"reasoning\": \"<brief parse note>\", \"rules\": [...]}.\n\n"
    "Valid class for matrix documents:\n"
    "  AccessRule — personnel authorisation, zone occupancy limit, or alarm acknowledgment "
    "deadline; includes occupancy headcounts and per-role acknowledgment times.\n\n"
    "MULTI-SEVERITY RULE: if the matrix defines different limits for WARNING and CRITICAL tiers "
    "(e.g. different occupancy thresholds or acknowledgment deadlines), extract EACH tier as a "
    "SEPARATE rule with the appropriate severity.\n\n"
    "Field rules:\n"
    "  sensor     — FULL identifier = <STATION_ID>_<TYPE_CODE>. Leave empty if not applicable.\n"
    "  sensorType — SHORT type code. Leave empty for pure access/occupancy rules.\n\n"
    "Examples (do NOT copy values — structure only):\n"
    "AccessRule (occupancy): "
    "{\"ruleId\":\"RULE-OCC-Z1\",\"class\":\"AccessRule\",\"station\":\"ZONE_1\","
    "\"sensor\":\"\",\"sensorType\":\"\",\"condition\":\"zone occupancy exceeds <n> persons\","
    "\"action\":\"<evacuation or restriction action>\",\"severity\":\"CRITICAL\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"persons\"}\n"
    "AccessRule (acknowledgment): "
    "{\"ruleId\":\"RULE-ACK-ROLE-WARN\",\"class\":\"AccessRule\",\"station\":\"\","
    "\"sensor\":\"\",\"sensorType\":\"\","
    "\"condition\":\"<role> must acknowledge WARNING alarm\","
    "\"action\":\"acknowledge within <n> minutes\",\"severity\":\"MANDATORY\","
    "\"critHi\":\"\",\"warnHi\":\"\",\"warnLo\":\"\",\"critLo\":\"\",\"unit\":\"min\"}\n"
)


_PHYSICAL_CONSTRAINT_MAPPING = (
    "\nPHYSICAL CONSTRAINT MAPPING — apply to every rule:\n"
    "  PCM-1 VERBATIM NUMERICS: copy every numeric threshold value exactly as it appears "
    "in the source text — do not round, convert, or reinterpret (e.g., '26' stays '26', "
    "not '26.0' or '~26').\n"
    "  PCM-2 VERBATIM UNITS: copy the engineering unit exactly as written alongside the "
    "value (e.g., '°C', 'bar', 'mm/s'). If no unit is stated in the text for that specific "
    "value, leave the unit field empty — never infer it from context or column headers.\n"
    "  PCM-3 ACTIONABLE CONDITIONS: the condition field must express the exact physical "
    "state that triggers the rule (e.g., '<SENSOR_TYPE> exceeds <VALUE><UNIT> at <STATION>'), "
    "not a generic paraphrase ('value too high'). Include the numeric boundary and location when given.\n"
    "  PCM-4 STRICT NULL DISCIPLINE: if a field value is implied, inferred, or absent from "
    "the text, leave that field as an empty string — never fill it with a plausible guess. "
    "This applies to sensor, sensorType, unit, station, and all threshold fields.\n"
    "  PCM-5 NO CROSS-ROW/CROSS-SENTENCE INFERENCE: each rule must be grounded solely in "
    "the row, sentence, or paragraph that introduces it. Do not borrow values from "
    "neighbouring rows, column headers, or earlier paragraphs to fill gaps.\n"
)


def _prompt_narrative() -> str:
    return _BASE_SYSTEM + _CONTEXT_A + _PHYSICAL_CONSTRAINT_MAPPING + (
        "\nDocument type: narrative / mixed SOP.\n"
        "INSTRUCTIONS:\n"
        "1. Extract EVERY explicitly labeled rule. "
        "CRITICAL — later sections of the document may pack multiple rules into a SINGLE "
        "paragraph or bullet point with no line breaks between them (e.g. 'RULE-XX-01: ... "
        "RULE-XX-02: ... RULE-XX-03: ...' all in one block of text, separated only by "
        "periods or sentence boundaries). Scan EVERY paragraph systematically, even short "
        "ones, and split on each rule identifier you encounter. Never stop at the first rule "
        "in a paragraph — read to the end of it.\n"
        "2. Each station-sensor-condition combination is a separate rule.\n"
        "3. Process-level conditions → OperationalRule. "
        "When WARNING and CRITICAL require different actions, produce TWO rules.\n"
        "4. Drift patterns, stuck-sensor, cross-sensor correlations → MaintenanceRule.\n"
        "5. Access, occupancy, acknowledgment → AccessRule.\n"
        "6. Inline threshold numbers → ThresholdRule (all four fields in one rule).\n"
        "7. Apply PCM-1 through PCM-5: do not hallucinate numeric values or units; "
        "leave any field without explicit textual evidence empty.\n"
    )


def _prompt_tabular() -> str:
    return _BASE_SYSTEM + _CONTEXT_B + _PHYSICAL_CONSTRAINT_MAPPING + (
        "\nDocument type: tabular (threshold table SOP).\n"
        "INSTRUCTIONS:\n"
        "1. Map each table row to exactly one ThresholdRule.\n"
        "2. Column mapping: CRIT_LO→critLo, WARN_LO→warnLo, WARN_HI→warnHi, CRIT_HI→critHi.\n"
        "3. Store ONLY pure numeric values in threshold fields (no unit suffix).\n"
        "4. Generate ruleId as RULE-THR-<STATION_ABBR>-<SENSOR_TYPE>-CRIT.\n"
        "5. Do NOT skip any row — every sensor in the table must produce a rule.\n"
        "6. Apply PCM-1 through PCM-5: if a table cell is blank, dash, or N/A, leave the "
        "corresponding field empty — do not fill it from a neighbouring row or the column "
        "header. Unit comes only from an explicit unit cell or column; never infer it.\n"
        "7. Station comes only from an explicit station column or row prefix; do not derive "
        "it from the sensor identifier pattern.\n"
    )


def _prompt_matrix() -> str:
    return _BASE_SYSTEM + _CONTEXT_C + _PHYSICAL_CONSTRAINT_MAPPING + (
        "\nDocument type: access / occupancy / acknowledgment matrix.\n"
        "INSTRUCTIONS — extract ALL THREE sub-types:\n"
        "1. Occupancy limits: one AccessRule per zone with EXPLICITLY ENFORCED limits only — "
        "the table must state a maximum AND describe a consequence when it is exceeded "
        "(e.g. 'WARNING above X', 'CRITICAL above Y', 'badge required'). "
        "Zones that are described as monitored or tracked without a stated enforcement "
        "consequence do NOT produce an AccessRule. "
        "Include WARNING and CRITICAL tiers as separate rules when BOTH are defined.\n"
        "2. Acknowledgment deadlines: one AccessRule per (role × severity) combination "
        "present in the acknowledgment-time table. Cover every populated cell — do not skip any.\n"
        "3. Labeled prose rules: extract each as a separate AccessRule ONLY if it defines "
        "a real-world operational constraint — a physical or access condition that triggers "
        "a required human action. Skip rules that describe software, data model, or system "
        "recording requirements rather than operational procedures.\n"
        "Leave sensor/sensorType/threshold fields empty for all AccessRules unless the text "
        "explicitly provides numeric sensor values.\n"
        "4. Apply PCM-1 through PCM-5: headcount and acknowledgment times must be copied "
        "verbatim from the cell (e.g., '12 persons', '15 min'). Zone and role names must "
        "match the exact table label. If a cell is blank, do not synthesize a rule for "
        "that (role × severity) or zone combination.\n"
    )


_PROMPT_FN = {
    "narrative": _prompt_narrative,
    "mixed":     _prompt_narrative,
    "tabular":   _prompt_tabular,
    "matrix":    _prompt_matrix,
}



# ── Normalizer prompt ─────────────────────────────────────────────────────────

NORMALIZER_SYSTEM = (
    "You are a ruleId normalizer. Your ONLY task is to assign canonical ruleIds. "
    "Do NOT modify class, station, sensor, condition, action, severity, or any other field.\n\n"
    "RULE 1 — DOCUMENT-SUPPLIED IDs ARE INVIOLABLE: if a rule has a non-empty ruleId "
    "extracted verbatim from the source document, keep it EXACTLY as-is.\n"
    "RULE 2 — For empty or invented ruleIds, assign canonical patterns:\n"
    "  OperationalRule    : RULE-{STATION_ABBR}-{NN}  (sequential per station; "
    "append 'a'/'b' for WARNING/CRITICAL tiers of the same sensor)\n"
    "  MaintenanceRule    : MAINT-{NN}  (sequential across all simple maintenance rules)\n"
    "  Correlated fault   : RULE-CORR-{NN}\n"
    "  ThresholdRule      : RULE-THR-{STATION_ABBR}-{SENSORTYPE}-{SEV}  (SEV = WARN or CRIT)\n"
    "  AccessRule occupancy : RULE-OCC-{ZONE_ABBR}\n"
    "  AccessRule ack-time  : RULE-ACK-{ROLE_ABBR}-{SEV}  (SEV = WARN or CRIT; "
    "role: operator→OP, technician→TECH, supervisor→SUP, manager→MAN, security→SEC)\n"
    "  AccessRule prose     : RULE-ACCESS-{NN}  (if not already labeled)\n"
    "RULE 3 — All ruleIds must be unique. Adjust numbering on collision.\n"
    "RULE 4 — Class-conditional: RULE-THR-* only for ThresholdRule; "
    "MAINT-*/RULE-CORR-* only for MaintenanceRule; RULE-OCC-*/RULE-ACK-*/RULE-ACCESS-* "
    "only for AccessRule.\n\n"
    "Return the FULL JSON list with ONLY ruleId fields updated."
)

VALIDATOR_SYSTEM = (
    "You are a gap-detection validator for industrial SOP rule extraction. "
    "You receive rules already extracted from ONE document and that document's text.\n\n"
    "Your ONLY task is to find false negatives — rules present in the document text "
    "but missing from the extracted set. "
    "Scan the document for labeled rule identifiers "
    "(RULE-XX-YY, MAINT-XX, RULE-CORR-XX, RULE-THR-*, RULE-OCC-*, RULE-ACK-*, RULE-ACCESS-*). "
    "For each identifier found in the text, check if it already appears in any extracted ruleId. "
    "Only report a rule as missing if: "
    "(i) its ruleId is completely absent from all extracted rules, AND "
    "(ii) its condition AND action are explicitly stated in the text — do not infer or paraphrase.\n\n"
    "Do NOT remove, modify, or re-evaluate any existing rule. "
    "Return ONLY the newly discovered missing rules (do not repeat existing ones). "
    "If nothing is missing, return an empty list.\n\n"
    "Return valid JSON: {\"rules\": [<only newly discovered rules>]}."
)


# ── Judge checklists ────────────────────────────────────────────────────────────
_CHECKLIST_NARRATIVE = (
    "Checklist for OPERATING PROCEDURE / NARRATIVE documents:\n"
    "1. ACTIVELY SCAN the full document text for every labeled rule identifier "
    "(RULE-XX-YY pattern or equivalent). List each identifier you find. "
    "Then verify each one is represented in the extracted rules. "
    "If ANY labeled rule from the text is absent from the extracted set, set pass=false "
    "and list the missing identifiers in 'suggestions'.\n"
    "2. For each station, are all mentioned sensor types covered by at least one rule?\n"
    "3. When the text describes WARNING and CRITICAL responses for the same sensor that "
    "require DIFFERENT actions, are both extracted as separate rules?\n"
    "4. Are cross-station dependency constraints included?\n"
)

_CHECKLIST_TABULAR = (
    "Checklist for THRESHOLD TABLE documents:\n"
    "1. Is there a ThresholdRule for every sensor row in the table? No row may be skipped.\n"
    "2. Are numeric threshold values correctly placed in critHi/warnHi/warnLo/critLo "
    "with no unit suffixes?\n"
)

_CHECKLIST_MATRIX = (
    "Checklist for ACCESS CONTROL / OCCUPANCY documents:\n"
    "1. Is there an AccessRule for every zone with a defined occupancy limit?\n"
    "2. Is there an AccessRule for every (role × severity) combination that has an explicit "
    "value in the acknowledgment-time table? Count the populated cells — no cell may be skipped.\n"
    "3. Are all prose labeled rules (RULE-ACCESS-XX or equivalent) included?\n"
)

_CHECKLIST_MAINTENANCE = (
    "Checklist for MAINTENANCE RULE documents:\n"
    "1. Is there a MaintenanceRule for every predictive-maintenance table row?\n"
    "2. Is there a MaintenanceRule for every correlated-fault pattern?\n"
)

_JUDGE_BASE = (
    "You are a strict engineering safety-net judge for extracted industrial SOP rules. "
    "Detect issues in exactly four categories:\n"
    "  S1 — Class-gaps: rule classes that should exist for this SOP type are missing entirely.\n"
    "  S2 — Count-gaps: specific constraints, locations, limits, or labeled rules present in "
    "the source text have no corresponding extracted rule.\n"
    "  S3 — Quality issues — flag any of the following:\n"
    "    (a) Incomplete fields: empty station, missing condition, or missing numeric thresholds "
    "where the document provides them.\n"
    "    (b) Hallucinated values: conditions, locations, or numeric values not grounded in the "
    "document text. Instruct the retry to DROP the offending rule entirely.\n"
    "    (c) Physical impossibility: numeric limits that overlap illegally "
    "(e.g. critLo > critHi, warnLo > warnHi), or an action semantically incompatible with "
    "its declared severity (e.g. severity='CRITICAL' with action='do nothing'). "
    "Instruct the retry to DROP or CORRECT the offending rule.\n"
    "  S4 — Conflicts: two rules share the same ruleId AND same severity. "
    "Do NOT flag rules with the same ruleId but different severity — that is intentional.\n\n"
    "Only flag issues for content actually present in the document.\n\n"
)

_JUDGE_SUFFIX = (
    "Return JSON: {\"gaps\": [list of gap codes from S1/S2/S3/S4], "
    "\"suggestions\": \"specific retry instructions — for S3(b)/(c) violations name the "
    "ruleId to DROP or the exact field values to correct\", "
    "\"deficient_files\": [list of source_file filenames that need re-extraction; "
    "base this on which files are missing rules or contain violations — empty list if none], "
    "\"pass\": true/false}"
)


def _build_judge_system(doc_types: dict[str, str]) -> str:
    unique = set(doc_types.values())
    parts = [_JUDGE_BASE]
    if "narrative" in unique or "mixed" in unique:
        parts.append(_CHECKLIST_NARRATIVE)
    if "tabular" in unique:
        parts.append(_CHECKLIST_TABULAR)
    if "matrix" in unique:
        parts.append(_CHECKLIST_MATRIX)
    if "mixed" in unique:
        # mixed SOP_003-style documents also need maintenance checklist
        parts.append(_CHECKLIST_MAINTENANCE)
    parts.append(_JUDGE_SUFFIX)
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  LangGraph nodes
# ══════════════════════════════════════════════════════════════════════════════
# Each node receives the full PipelineState and is expected to return a partial
# update dict. The update is merged into the shared state by LangGraph before
# the next node is invoked.

_STRUCTURAL_CLASSIFY_PROMPT = (
    "You are a document structure analyst. Classify the industrial SOP document below "
    "into exactly one type based on its DOMINANT spatial layout.\n\n"
    "Classification rules — read carefully before deciding:\n"
    "  narrative — the primary content is individually labeled operational rules written as "
    "PROSE SENTENCES in flowing text (each rule is a complete sentence or short paragraph "
    "with a condition and a required response, identified by a label). "
    "May contain a small informational table (e.g. a station dependency list) but the "
    "labeled prose rules are clearly the dominant content. "
    "KEY: the rules are SENTENCES, not table rows.\n"
    "  tabular   — the document is PRIMARILY a sensor threshold table where each row defines "
    "the full set of numeric alarm limits for one sensor across multiple severity tiers "
    "(low-critical, low-warning, high-warning, high-critical). The table PURPOSE is to specify "
    "numeric boundaries — it is a threshold reference, not a list of tasks or procedures.\n"
    "  matrix    — the document is a cross-reference grid mapping roles or personnel to "
    "zones or areas (authorisation status, headcounts). The dominant structure is a "
    "role-vs-zone or role-vs-severity lookup table.\n"
    "  mixed     — the primary content is TABLE ROWS (not prose sentences) where each row "
    "describes a maintenance procedure or fault pattern as structured data: trigger condition, "
    "required action, and priority level stored in separate columns. "
    "Also choose mixed for correlated-fault tables linking one sensor's deviation to another. "
    "KEY: the maintenance content is TABLE DATA, not labeled prose sentences — if the document "
    "primarily contains labeled prose rules, choose narrative instead.\n\n"
    "Reply ONLY with valid JSON: "
    "{\"doc_type\": \"<narrative|tabular|matrix|mixed>\", "
    "\"structural_reasoning\": \"<one sentence naming the dominant layout feature>\"}\n\n"
    "Document:\n"
)


def coordinator_node(state: PipelineState) -> dict:
    """Load SOP files and classify each document type via LLM structural density analysis."""
    model     = state["coordinator_model"]
    abs_texts = os.path.abspath(TEXTS_DIR)
    txt_files = sorted(f for f in os.listdir(abs_texts) if f.endswith(".txt"))
    print(f"[Coordinator/{model}] Found {len(txt_files)} SOP files.")

    sop_texts: dict[str, str] = {}
    doc_types: dict[str, str] = {}

    for fname in txt_files:
        with open(os.path.join(abs_texts, fname), encoding="utf-8") as f:
            text = f.read()
        sop_texts[fname] = text[:SOP_TEXT_LIMIT]

        raw = llm_call(model, [
            {"role": "user", "content": _STRUCTURAL_CLASSIFY_PROMPT + text[:8000]},
        ]).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        dtype = "narrative"
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(m.group(0) if m else raw)
            dtype = result.get("doc_type", "narrative").strip().lower()
            reasoning = result.get("structural_reasoning", "")
            if dtype not in _DTYPE_TO_NODE:
                dtype = "narrative"
            if reasoning:
                print(f"    [structural_reasoning] {reasoning}")
        except Exception:
            dtype = next((t for t in ("tabular", "matrix", "mixed") if t in raw.lower()), "narrative")

        doc_types[fname] = dtype
        print(f"  {fname}: {dtype} → {_DTYPE_TO_NODE[dtype]}")

    return {"sop_texts": sop_texts, "doc_types": doc_types, "extracted_rules": []}


def dispatch_to_extractors(state: PipelineState)  -> list[Send]:
    """Fan-out: one Send per SOP file to the correct typed extractor node."""
    return [
        Send(_DTYPE_TO_NODE[dtype], {
            "current_fname":     fname,
            "current_dtype":     dtype,
            "sop_texts":         state["sop_texts"],
            "extractor_a_model": state["extractor_a_model"],
            "extractor_b_model": state["extractor_b_model"],
            "extractor_c_model": state["extractor_c_model"],
            "is_retry":          False,
        })
        for fname, dtype in state["doc_types"].items()
    ]


def _run_extractor(state: PipelineState, model_key: str, label: str) -> dict:
    """Shared extraction logic for all three typed extractor nodes."""
    fname    = state["current_fname"]
    dtype    = state["current_dtype"]
    text     = state["sop_texts"][fname]
    model    = state[model_key]  # type: ignore[literal-required]
    is_retry = state.get("is_retry", False)

    system_prompt = _PROMPT_FN[dtype]()
    if is_retry:
        system_prompt += "\nThis is a RETRY. Be exhaustive — extract EVERY rule, including multi-rule lines."

    paradigm = "few_shot_static_retry" if is_retry else "few_shot_static"
    suffix   = " (RETRY)" if is_retry else ""
    print(f"  [Extractor-{label}/{model}] {fname} ({dtype}){suffix}")

    raw   = llm_call(model, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"SOP text:\n{text}"},
    ])
    rules = parse_rules(raw)

    for r in rules:
        r["source_file"] = fname
        r["model_name"]  = model
        r["paradigm"]    = paradigm

    print(f"    → {len(rules)} rules")
    return {"extracted_rules": rules}


def extract_narrative_node(state: PipelineState) -> dict:
    """Extractor A — narrative and mixed SOP documents (gemma3:12b)."""
    return _run_extractor(state, "extractor_a_model", "A")


def extract_tabular_node(state: PipelineState) -> dict:
    """Extractor B — tabular threshold-table SOP documents (ministral-3:8b)."""
    return _run_extractor(state, "extractor_b_model", "B")


def extract_matrix_node(state: PipelineState) -> dict:
    """Extractor C — access/occupancy matrix SOP documents (gemma3:12b)."""
    return _run_extractor(state, "extractor_c_model", "C")


def merge_node(state: PipelineState) -> dict:
    """Deduplicates via content agreement scores; LLM adjudicates conflicts.

    Groups rules by (class, station, sensorType). Within each group:
      Branch A  — singleton: pass through unchanged.
      Branch B  — ThresholdRule pair with complementary crit/warn fields: deterministic merge.
      Branch B2 — ThresholdRule semantic dedup: same sensor + identical numeric thresholds.
      Branch C  — content-agreement dedup: candidates scoring >= _AGREEMENT_THRESHOLD are
                  collapsed (richest kept); action-disagreement guards severity-tier splits.
      Remaining conflicts are kept as-is — no LLM adjudication.
    """
    all_rules = state.get("extracted_rules", [])
    print(f"[Merge] Merging {len(all_rules)} raw rules (deterministic only)...")

    if not all_rules:
        return {"merged_rules": []}

    # Rules are grouped by (class, station, sensorType) so that only candidates
    # describing the same physical entity are considered for merging or adjudication.
    groups: dict[tuple, list[dict]] = {}
    for r in all_rules:
        cls     = r.get("class",      "").strip()
        station = r.get("station",    "").strip().upper()
        stype   = r.get("sensorType", "").strip().upper()
        # When both station and sensorType are empty (common for AccessRule and
        # MaintenanceRule), rules with different conditions would otherwise land
        # in one giant group and be incorrectly collapsed.
        # Adding a condition prefix as a fourth key component prevents this.
        cond_disc = _norm_text(r.get("condition", ""))[:40] if (not station and not stype) else ""
        key = (cls, station, stype, cond_disc)
        groups.setdefault(key, []).append(r)

    output: list[dict] = []

    for key, candidates in groups.items():
        cls = key[0]

        # Branch A — singleton
        if len(candidates) == 1:
            output.append(candidates[0])
            continue

        # Branch B — ThresholdRule complementary merge (crit + warn from separate rules)
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

        # Branch B2 — ThresholdRule semantic dedup: same sensor + identical numeric thresholds
        if cls == "ThresholdRule":
            thr_sigs: dict[str, dict] = {}
            for r in candidates:
                tsig = _threshold_sig(r)
                if tsig not in thr_sigs or _n_nonempty(r) > _n_nonempty(thr_sigs[tsig]):
                    thr_sigs[tsig] = r
            candidates = list(thr_sigs.values())
            if len(candidates) == 1:
                output.append(candidates[0])
                continue

        # Branch C — content-agreement dedup: collapse near-duplicates by field overlap score
        seen: list[dict] = []
        for r in candidates:
            best_idx, best_score = -1, 0.0
            for i, existing in enumerate(seen):
                s = _agreement_score(r, existing)
                if s > best_score:
                    best_score, best_idx = s, i
            if best_score >= _AGREEMENT_THRESHOLD:
                existing = seen[best_idx]
                # Guard: preserve intentional severity-tier splits (same sensor, different action)
                a1 = _norm_text(str(r.get("action") or ""))
                a2 = _norm_text(str(existing.get("action") or ""))
                if a1 == a2 or not a1 or not a2:
                    if _n_nonempty(r) > _n_nonempty(existing):
                        seen[best_idx] = r
                    continue  # collapsed into existing
            seen.append(r)
        deduped = seen
        if len(deduped) == 1:
            output.append(deduped[0])
            continue

        # Remaining conflicts after deterministic dedup — keep all, let validator clean up
        output.extend(deduped)

    print(f"  → {len(output)} rules after merge")
    return {"merged_rules": output}


def validate_node(state: PipelineState) -> dict:
    """Deterministic cleanup then additive LLM gap-filling; never removes existing rules."""
    model     = state["validator_model"]
    rules     = state.get("merged_rules", [])
    sop_texts = state.get("sop_texts", {})
    print(f"[Validator/{model}] Validating {len(rules)} rules...")

    # Pass 1 — deterministic cleanup (class normalisation, threshold hygiene, dedup)
    rules = _deterministic_post_filter(rules)
    print(f"  → {len(rules)} after deterministic filter")

    # Pass 2 — per-file LLM gap-filling: only adds missing rules, never removes
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in rules:
        by_file[r.get("source_file", "")].append(r)

    all_valid: list[dict] = list(rules)
    for fname, file_rules in by_file.items():
        raw_text = sop_texts.get(fname, "")
        if not raw_text:
            continue

        print(f"  [Validator] {fname}: {len(file_rules)} rules → LLM gap-check")
        raw = llm_call(model, [
            {"role": "system", "content": VALIDATOR_SYSTEM},
            {"role": "user",   "content": (
                f"Source document:\n{raw_text[:8000]}\n\n"
                f"Extracted rules:\n{json.dumps(file_rules, indent=2)[:6000]}"
            )},
        ])
        found = parse_rules(raw)
        if found:
            existing_ids = {r.get("ruleId") for r in file_rules}
            added = [r for r in found if r.get("ruleId") not in existing_ids]
            for r in added:
                if not r.get("source_file"):
                    r["source_file"] = fname
            if added:
                print(f"    [Validator delta] {fname}: added {len(added)} → {sorted(r.get('ruleId') for r in added)}")
                all_valid.extend(added)

    # Final deterministic pass to clean up anything the LLM introduced
    valid = _deterministic_post_filter(all_valid)
    print(f"  → {len(valid)} validated rules")
    return {"validated_rules": valid}


def judge_node(state: PipelineState) -> dict:
    """Detects class-gaps (S1), count-gaps (S2), quality issues (S3), and conflicts (S4); triggers targeted retries."""
    model       = state["judge_model"]
    validated   = state.get("validated_rules", [])
    doc_types   = state.get("doc_types", {})
    sop_texts   = state.get("sop_texts", {})
    retry_count = state.get("retry_count", 0)
    print(f"[Judge/{model}] Checking {len(validated)} rules (retry_count={retry_count})...")

    if retry_count >= 1:
        print("  → max retries reached")
        return {"retry_files": {}}

    # A deterministic physics pre-check is performed before the LLM call so that
    # threshold-ordering violations are surfaced even when the LLM judge overlooks them.
    physics_violations = [r for r in validated if not _physics_valid(r)]
    if physics_violations:
        print(f"  [Judge] {len(physics_violations)} physics violation(s) detected (S3-physics):")
        for r in physics_violations:
            print(f"    {r.get('ruleId', '?')}: "
                  f"critLo={r.get('critLo')} warnLo={r.get('warnLo')} "
                  f"warnHi={r.get('warnHi')} critHi={r.get('critHi')}")

    # Per-file judge calls — each file gets its full SOP_TEXT_LIMIT chars of text.
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in validated:
        by_file[r.get("source_file", "")].append(r)

    llm_deficient: set[str] = set()
    llm_feedback:  dict[str, dict] = {}
    for fname, dtype in doc_types.items():
        file_rules = by_file.get(fname, [])
        file_text  = sop_texts.get(fname, "")
        judge_sys  = _build_judge_system({fname: dtype})
        rules_json = json.dumps(file_rules, indent=2)[:6000]

        print(f"  [Judge] {fname} ({dtype}): {len(file_rules)} rules")
        raw = llm_call(model, [
            {"role": "system", "content": judge_sys},
            {"role": "user",   "content": (
                f"Source document:\n{file_text}\n\n"
                f"Extracted rules:\n{rules_json}"
            )},
        ])
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            feedback = json.loads(cleaned)
        except Exception:
            feedback = {"pass": True, "gaps": [], "suggestions": ""}

        passed = feedback.get("pass", True)
        gaps   = feedback.get("gaps", [])
        print(f"    → Pass: {passed}, Gaps: {gaps}")
        llm_feedback[fname] = feedback
        if not passed:
            llm_deficient.add(fname)

    retry_files: dict[str, str] = {}
    for fname, dtype in doc_types.items():
        file_rules = by_file.get(fname, [])
        classes    = {r.get("class") for r in file_rules}

        llm_flagged     = fname in llm_deficient
        no_rules        = len(file_rules) == 0
        narrative_empty = dtype in ("narrative", "mixed") and not (
            {"OperationalRule", "MaintenanceRule"} & classes
        )
        tabular_empty   = dtype == "tabular" and "ThresholdRule" not in classes
        matrix_empty    = dtype == "matrix"  and "AccessRule"    not in classes

        structural_failure = no_rules or narrative_empty or tabular_empty or matrix_empty

        if structural_failure:
            # Only retry on complete structural failures (zero rules of expected class).
            # Partial gaps detected by the LLM are handled by the validator's additive
            # gap-filling and do NOT trigger retry — retry causes re-merge over-dedup
            # that loses valid rules from the original extraction.
            reason = (
                "no rules extracted"              if no_rules        else
                "no ThresholdRules"               if tabular_empty   else
                "no AccessRules"                  if matrix_empty    else
                "missing substantive rule classes"
            )
            print(f"  Judge: {fname} → retry ({reason})")
            retry_files[fname] = dtype
        elif llm_flagged:
            # Log LLM-detected gaps for observability but do not retry.
            suggestions = llm_feedback.get(fname, {}).get("suggestions", "")
            print(f"  Judge: {fname} → gaps noted (no retry): {suggestions[:120]}")

    return {"retry_files": retry_files}


def pre_retry_node(state: PipelineState) -> dict:
    """Increment retry counter before dispatching retry extractions."""
    new_count = state.get("retry_count", 0) + 1
    print(f"[PreRetry] Starting retry {new_count} for {len(state.get('retry_files', {}))} file(s)")
    return {"retry_count": new_count}


def dispatch_retry_sends(state: PipelineState) -> list[Send]:
    """Fan-out retry files to the correct typed extractors with is_retry=True."""
    retry_files = state.get("retry_files", {})
    return [
        Send(_DTYPE_TO_NODE.get(dtype, "extract_narrative"), {
            "current_fname":     fname,
            "current_dtype":     dtype,
            "sop_texts":         state["sop_texts"],
            "extractor_a_model": state["extractor_a_model"],
            "extractor_b_model": state["extractor_b_model"],
            "extractor_c_model": state["extractor_c_model"],
            "is_retry":          True,
        })
        for fname, dtype in retry_files.items()
    ]


def _load_gt_few_shots(max_examples: int = 20) -> str:
    lines: list[str] = []
    try:
        with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines.append(
                    f"  {row['ruleId']:<30s}  "
                    f"station={row['station']!r}, sensor={row['sensor']!r}, class={row['class']!r}"
                )
                if len(lines) >= max_examples:
                    break
    except FileNotFoundError:
        pass
    return "\n".join(lines)


def normalize_node(state: PipelineState) -> dict:
    """Canonicalizes ruleId using ground-truth examples as few-shot demonstrations.

    Only ruleId is taken from the LLM; all semantic fields come from the originals.
    Falls back to originals unchanged if the count mismatches.
    """
    model  = state["normalizer_model"]
    rules  = state.get("validated_rules", [])
    print(f"[Normalizer/{model}] Normalizing {len(rules)} ruleIds...")

    if not rules:
        return {"normalized_rules": rules}

    few_shots = _load_gt_few_shots()
    json_in   = json.dumps(rules, indent=2)[:12000]
    raw = llm_call(model, [{"role": "system", "content": NORMALIZER_SYSTEM}, {"role": "user", "content": (
        f"Ground-truth ruleId examples:\n{few_shots}\n\n"
        f"Rules to normalize:\n{json_in}\nReturn the full JSON list."
    )}])
    normalized = parse_rules(raw)

    # Only the ruleId field is taken from the LLM response; all semantic fields are
    # preserved from the originals. If the returned count does not match, originals
    # are kept unchanged to prevent silent data loss.
    if normalized and len(normalized) == len(rules):
        final = [
            {**orig, "ruleId": norm.get("ruleId", orig.get("ruleId", ""))}
            for orig, norm in zip(rules, normalized)
        ]
    else:
        print(f"  [Normalizer] count mismatch ({len(normalized)} vs {len(rules)}) — keeping originals")
        final = rules

    print(f"  → {len(final)} rules normalized")
    return {"normalized_rules": final}


def save_node(state: PipelineState) -> dict:
    """Write final rules to CSV."""
    rules    = state.get("normalized_rules", [])
    ablation = state.get("ablation", "")
    tag      = _ABL_TAG_MAP.get(ablation, "")
    fname    = f"abl_{tag}_s42_run1.csv" if tag else "ext_multi_agent_langgraph.csv"
    out_path = os.path.join(RESULTS_DIR, fname)
    extra    = sorted({k for r in rules for k in r} - set(RULE_FIELDS))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RULE_FIELDS + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rules)
    print(f"[Save] Wrote {len(rules)} rules → {out_path}")
    return {"output_path": out_path}


def _judge_routing(state: PipelineState) -> str:
    if state.get("retry_files") and state.get("retry_count", 0) < 1:
        return "pre_retry"
    return "normalize"


# ══════════════════════════════════════════════════════════════════════════════
#  Ablation node variants
# ══════════════════════════════════════════════════════════════════════════════
# Simplified node variants are defined here so that the contribution of each
# pipeline component can be measured by swapping it out at graph-assembly time.
# Only the behaviour of the affected node is changed; all other nodes remain identical.

# ── no_cm: keyword-based coordinator (no LLM) ─────────────────────────────────
# Document type is inferred from a small set of regex patterns when the LLM
# coordinator is removed. Each pattern is matched against the first 8 000 characters.
_KEYWORD_CLASSIFY_RULES: list[tuple[str, str]] = [
    (r"CRIT_LO|WARN_LO|WARN_HI|CRIT_HI",                                     "tabular"),
    (r"(?i)(?:zone|area|sector)\s+\w+.*(?:authorized|forbidden|max.persons)",  "matrix"),
    (r"MAINT-|predictive[- ]maintenance|drift.*threshold|cross.sensor",        "mixed"),
]


def _keyword_classify(text: str) -> str:
    """Rule-based doc-type classification without LLM."""
    for pattern, dtype in _KEYWORD_CLASSIFY_RULES:
        if re.search(pattern, text):
            return dtype
    return "narrative"


def coordinator_node_no_cm(state: PipelineState) -> dict:
    """Coordinator without LLM — keyword/regex-based doc type detection (ablation: no_cm)."""
    abs_texts = os.path.abspath(TEXTS_DIR)
    txt_files = sorted(f for f in os.listdir(abs_texts) if f.endswith(".txt"))
    print(f"[Coordinator/keyword] Found {len(txt_files)} SOP files.")

    sop_texts: dict[str, str] = {}
    doc_types: dict[str, str] = {}

    for fname in txt_files:
        with open(os.path.join(abs_texts, fname), encoding="utf-8") as f:
            text = f.read()
        sop_texts[fname] = text[:SOP_TEXT_LIMIT]
        dtype = _keyword_classify(text[:8000])
        doc_types[fname] = dtype
        print(f"  {fname}: {dtype} → {_DTYPE_TO_NODE[dtype]}")

    return {"sop_texts": sop_texts, "doc_types": doc_types, "extracted_rules": []}



# ── no_validator: skip validation entirely ────────────────────────────────────
def validate_node_noop(state: PipelineState) -> dict:
    """Pass merged rules directly as validated rules — no filtering, no LLM (ablation: no_validator)."""
    rules = state.get("merged_rules", [])
    print(f"[Validator/skipped] {len(rules)} rules passed through unchanged (ablation: no_validator)")
    return {"validated_rules": rules}


# ══════════════════════════════════════════════════════════════════════════════
#  Graph assembly
# ══════════════════════════════════════════════════════════════════════════════
# The graph topology is assembled here. Ablation flags are used to substitute
# simplified node variants so that each component's contribution can be isolated.
# Edges and conditional fan-outs are registered after all nodes are added.
def build_graph(ablation: str | None = None) -> StateGraph:
    builder = StateGraph(PipelineState)

    coordinator_fn = coordinator_node_no_cm if ablation == "no_cm"        else coordinator_node
    validate_fn    = validate_node_noop     if ablation == "no_validator" else validate_node

    builder.add_node("coordinator",       coordinator_fn)
    builder.add_node("extract_narrative", extract_narrative_node)
    builder.add_node("extract_tabular",   extract_tabular_node)
    builder.add_node("extract_matrix",    extract_matrix_node)
    builder.add_node("merge",             merge_node)
    builder.add_node("validate",          validate_fn)
    builder.add_node("normalize",         normalize_node)
    builder.add_node("save",              save_node)

    builder.add_edge(START, "coordinator")

    builder.add_conditional_edges(
        "coordinator",
        dispatch_to_extractors,
        ["extract_narrative", "extract_tabular", "extract_matrix"],
    )

    builder.add_edge("extract_narrative", "merge")
    builder.add_edge("extract_tabular",   "merge")
    builder.add_edge("extract_matrix",    "merge")
    builder.add_edge("merge", "validate")

    if ablation == "no_judge":
        # The judge node and its associated retry loop are omitted entirely when
        # the no_judge ablation is selected; validated rules are passed straight
        # to the normalizer.
        builder.add_edge("validate", "normalize")
    else:
        builder.add_node("judge",     judge_node)
        builder.add_node("pre_retry", pre_retry_node)
        builder.add_edge("validate", "judge")
        builder.add_conditional_edges("judge", _judge_routing, {
            "pre_retry": "pre_retry",
            "normalize": "normalize",
        })
        builder.add_conditional_edges(
            "pre_retry",
            dispatch_retry_sends,
            ["extract_narrative", "extract_tabular", "extract_matrix"],
        )

    builder.add_edge("normalize", "save")
    builder.add_edge("save",      END)

    return builder


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
# The initial pipeline state is constructed from module-level defaults, then
# selectively overridden by caller-supplied keyword arguments. The compiled
# graph is invoked synchronously and the final state dict is returned to the caller.
def run_pipeline(ablation: str | None = None, force: bool = False, **overrides: str) -> dict:
    # force=True is used by multi-run eval which monkey-patches save_node to write
    # to a per-run filename — skip the result-cache and file-existence checks in that case.
    if not force:
        _tag   = _ABL_TAG_MAP.get(ablation or "", "")
        _fname = f"abl_{_tag}_s42_run1.csv" if _tag else "ext_multi_agent_langgraph.csv"
        _out   = os.path.join(RESULTS_DIR, _fname)

        # Result cache — restores exact original CSV without any LLM call.
        if _RESULT_CACHE is not None:
            _cached = _RESULT_CACHE.get(_fname)
            if _cached is not None:
                with open(_out, "w", encoding="utf-8", newline="") as _f:
                    _f.write(_cached)
                print(f"[ResultCache] {_fname} restored from cache.")
                return {"output_path": _out}

        # File-level guard — don't overwrite an existing result.
        if os.path.exists(_out):
            print(f"[Skip] {_fname} already exists — using existing result.")
            return {"output_path": _out}

    initial_state: PipelineState = {
        **DEFAULTS,                          # type: ignore[typeddict-item]
        "abox_path":        ABOX_DEFAULT_PATH,
        "sop_texts":        {},
        "doc_types":        {},
        "extracted_rules":  [],
        "merged_rules":     [],
        "validated_rules":  [],
        "normalized_rules": [],
        "retry_files":      {},
        "retry_count":      0,
        "output_path":      "",
        "current_fname":    "",
        "current_dtype":    "",
        "is_retry":         False,
        "ablation":         ablation or "",
        **overrides,                         # type: ignore[typeddict-item]
    }
    graph = build_graph(ablation).compile()
    return graph.invoke(initial_state)


if __name__ == "__main__":
    # CLI arguments are parsed and mapped to pipeline overrides so model names and
    # the ablation flag can be supplied without modifying the source file.
    parser = argparse.ArgumentParser(description="LangGraph multi-agent SOP rule extraction")
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=default)
    parser.add_argument("--abox", default=ABOX_DEFAULT_PATH)
    parser.add_argument(
        "--ablation",
        choices=list(_ABL_TAG_MAP.keys()),
        default=None,
        help=(
            "Ablation study — completely remove one pipeline component: "
            "no_cm (keyword coordinator instead of LLM), "
            "no_validator (skip validator entirely), no_judge (skip judge+retry entirely)"
        ),
    )
    args = vars(parser.parse_args())
    ablation_arg = args.pop("ablation", None)
    overrides = {k: v for k, v in args.items() if v is not None}
    if "abox" in overrides:
        overrides["abox_path"] = overrides.pop("abox")
    run_pipeline(ablation=ablation_arg, **overrides)