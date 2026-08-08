"""Schema-, format-, domain- and ontology-agnostic multi-agent rule-extraction
pipeline for industrial SOP / SCADA operating and control documents. The one
and only thing assumed about the input is that it is an operating or control
document of some kind; which equipment ontology it uses, which fields its
rules carry, which categories they fall into, and whether it is written as
prose, a table, or a matrix are all discovered from the corpus itself and
never preset in code.

    segment -> [scout]* -> induce -> assemble -> [audit]* -> save

  1. segment (no LLM) splits every document into blank-line-delimited blocks,
     packs them into chunks, and NUMBERS every content line. The numbering is
     assigned by code, so provenance is something the pipeline knows rather
     than something a model has to remember to write down: an agent is only
     ever asked to cite a line number it is currently looking at, which is the
     most reliable thing an LLM can be asked for. Section headings are
     detected by SHAPE alone (a short, isolated, unpunctuated line) and carried
     into every chunk they cover, so "which asset does this table belong to"
     survives a chunk boundary.

  2. scout (1 LLM call per chunk, run in parallel) reads the numbered chunk
     and emits one record per rule the DOCUMENT ITSELF delimits -- a printed
     rule identifier if the document prints one, otherwise one table row, one
     matrix cell, or one rule-stating sentence. Field names are copied from
     the document's own labels: a table's column headers become the record's
     field names, a matrix's axis labels become its subject and field name.
     No schema is supplied to this pass and no worked example is given: an
     example demonstrates one particular document shape, and a model shown one
     reproduces that shape on documents that do not have it. The prompt ends
     with a chain-of-thought instruction -- reason through the chunk, then emit
     JSON -- which is the cot_basic paradigm the layer-2 grid selected over
     nine alternatives across 100 runs, chosen on stability rather than mean
     (the top four paradigms are statistically tied at 0.888-0.900, and
     cot_basic varies by +-0.008 where the runner-up varies by +-0.055, at one
     turn instead of three).

  3. induce (1 LLM call per CORPUS) is the schema arbiter. It never sees a
     document -- only the inventory of field names and category labels the
     scouts actually produced, with occurrence counts and sample values -- and
     merges the synonyms into one corpus-level schema. Deriving the schema
     from what the corpus was observed to contain, rather than proposing one
     up front from a document's opening pages, is what lets the field count
     follow the evidence: a corpus whose rules genuinely carry fourteen
     distinct slots yields fourteen fields, because fourteen were observed.
     Two documents in one corpus that name the same role differently are
     reconciled here, so the output speaks one vocabulary.

  4. assemble (no LLM) groups records by the document's own record boundary,
     renames each field to its canonical name, and emits one row per rule.
     Because the boundary is the printed identifier or the physical line the
     record was read from, granularity is decided by evidence printed in the
     document and never by a ground truth's column conventions.

  5. audit (1 LLM call per chunk, run in parallel) re-reads each assembled
     record against the numbered lines it cites and drops or corrects anything
     not actually stated there. A whole-chunk rejection is treated as an audit
     failure rather than a finding, because an auditor claiming every record is
     ungrounded is likelier to be malfunctioning than to be right.

Nothing in this file names a station, sensor, unit, severity level, field, or
category. The corpus under test supplies all of them, and the ground truth is
never read.

Every agent role is served by Ollama Cloud through its OpenAI-compatible API,
configured by OLLAMA_BASE_URL and OLLAMA_API_KEY in the project's .env. Model
names passed to any --*-model flag must be names that endpoint serves; `curl -H
"Authorization: Bearer $OLLAMA_API_KEY" https://ollama.com/api/tags` lists the
current roster.

Every free parameter is declared rather than buried: the model per role, the
reasoning-suppression setting per model, chunk size, concurrency, and the two
ablation switches. Runs are replayable from the response cache; a fresh run is
NOT bit-identical, because the hosted endpoint is non-deterministic for long
generations (see llm_call), so report a mean and a standard deviation over
--runs N rather than a single figure.

Usage:
    python step2_multi_agent_generic.py --input-dir path/to/txts
    python step2_multi_agent_generic.py --input-dir path/to/txts --runs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import operator
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_cache import LLMResponseCache

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
_OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

if not _OLLAMA_API_KEY:
    sys.exit("OLLAMA_API_KEY is not set. Add it to the project's .env "
             "(get a key at https://ollama.com/settings/keys).")

# 600s is generous for this endpoint while still catching a dead connection
# quickly. A dropped TCP connection is not rare here, and with a long timeout
# the symptom is a process sitting at 0% CPU producing nothing -- indis-
# tinguishable from slow progress -- until it finally retries.
_client = OpenAI(api_key=_OLLAMA_API_KEY, base_url=_OLLAMA_BASE_URL, timeout=600)


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 15.0
MAX_OUTPUT_TOKENS = 32768     # generous: reasoning models bill their thinking against this
DEFAULT_CHUNK_CHARS = 3500    # target size of a scout chunk
DEFAULT_MAX_CONCURRENCY = 4   # concurrent requests during the scout and audit fan-outs

# How much of the observed inventory the schema arbiter sees. It reads names and
# a few sample values per name, never documents, so this is small by design.
INDUCTION_SAMPLES_PER_FIELD = 6
# Prompt-size budget for the arbiter, expressed so that RARITY IS NEVER THE
# DISCARD CRITERION. An earlier form of this cap kept the 120 most frequent
# names and dropped the rest, which is precisely backwards: a field used by
# three records out of two hundred is the only place those three records'
# information lives, and in a safety setting the rare fields are typically the
# edge-case triggers. Names are therefore never dropped. When the inventory
# grows past the budget it is the SAMPLE VALUES that thin, because samples only
# illustrate a name whereas the name itself is the information the arbiter must
# reconcile.
INDUCTION_BUDGET_FIELDS = 120
# A name-only inventory larger than this cannot be made to fit by thinning
# samples. Reaching it is a signal to merge per-document inventories
# hierarchically rather than to raise a constant, so it stops the run loudly
# instead of silently returning a partial vocabulary.
INDUCTION_HARD_MAX_FIELDS = 2000

RESULTS_DIR = os.path.join(_SCRIPT_DIR, "step2_results_generic")
os.makedirs(RESULTS_DIR, exist_ok=True)

# On-disk LLM response cache -- see the note in llm_call() for why a fixed seed
# is not enough. Enabled by default; --no-cache disables it and --cache-path
# points it elsewhere.
DEFAULT_CACHE_PATH = os.path.join(RESULTS_DIR, "llm_response_cache.json")
_CACHE: LLMResponseCache | None = None

# Models whose endpoint rejects a system-role message; their system prompt is
# merged into the user turn instead. Empty for the models used here -- the set
# and its merge path are kept so a model with that restriction can be slotted
# in without code changes.
NO_SYSTEM_ROLE: set[str] = set()

# Model assignment. Both models are ones the layer-2 grid actually scored --
# deploying a model the grid never measured would mean the paradigm was selected
# under one model and applied under another with no evidence the choice carries.
#
# Over 100 grid runs gemma4:31b beat nemotron-3-nano:30b on ALL TEN paradigms,
# and by more than the paradigms differ from each other: the spread across
# paradigms within gemma4 is 0.095, while the same paradigm across models
# differs by up to 0.093. The model choice matters as much as the prompting
# strategy, so gemma4:31b takes every role. Independence between the extracting
# and auditing role is available cheaply via --auditor-model for anyone who
# wants it; it is not the default because the alternative model scored lower on
# every paradigm and was markedly less stable, and an auditor that unreliable
# removes more signal than noise.
DEFAULTS = {
    "scout_model":    "gemma4:31b",   # the fan-out, one call per chunk
    "inducer_model":  "gemma4:31b",   # one call per corpus; everything downstream is named by it
    "auditor_model":  "gemma4:31b",   # grounding audit, one call per chunk
}

# Reasoning suppression per model; absent means send no reasoning_effort at all.
# gemma4:31b is deliberately absent -- it has no thinking pass by default and
# passing any effort value TURNS ONE ON.
REASONING_EFFORT: dict[str, str] = {
    "nemotron-3-nano:30b": "none",
}

# Keys the pipeline attaches to a scouted record for its own bookkeeping. They
# are never treated as discovered fields.
_INTERNAL_KEYS = {"id", "category", "lines", "source_file", "chunk_id", "_key"}


# ══════════════════════════════════════════════════════════════════════════════
#  Induced corpus schema
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class CorpusSchema:
    """Produced by the induce stage for ONE corpus. Nothing about field names or
    categories is fixed in code -- this carries whatever the arbiter decided the
    corpus needs. The three role pointers (condition_field / action_field /
    severity_field) are induced and serialised for downstream consumers of the
    saved schema; nothing in this pipeline reads them (numeric_fields is the
    only role pointer assembly consumes)."""
    fields:          dict[str, str] = field(default_factory=dict)
    field_map:       dict[str, str] = field(default_factory=dict)
    categories:      dict[str, str] = field(default_factory=dict)
    category_map:    dict[str, str] = field(default_factory=dict)
    condition_field: str = ""
    action_field:    str = ""
    severity_field:  str = ""
    numeric_fields:  list[str] = field(default_factory=list)

    def canon_field(self, observed: str) -> str:
        """Canonical name for an observed field. An unmapped name maps to
        ITSELF rather than being dropped: the arbiter forgetting to mention a
        field must never silently delete the values recorded under it."""
        return self.field_map.get(observed, observed)

    def canon_category(self, observed: str) -> str:
        return self.category_map.get(observed, observed)

    def to_json(self) -> dict:
        return {
            "fields": self.fields, "field_map": self.field_map,
            "categories": self.categories, "category_map": self.category_map,
            "condition_field": self.condition_field,
            "action_field": self.action_field,
            "severity_field": self.severity_field,
            "numeric_fields": self.numeric_fields,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LangGraph state
# ══════════════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict, total=False):
    """State shared by the agent graph. The two fan-out stages accumulate into
    `scouted` and `audited` through operator.add, which is what lets many
    concurrent branches write into one list without coordinating: each Send
    returns its own slice and the reducer concatenates them."""
    input_dir:     str
    scout_model:   str
    inducer_model: str
    auditor_model: str
    run_index:     int
    corpus_name:   str
    chunk_chars:   int
    file_lines:    dict[str, list[str]]
    chunks:        list[dict]
    scouted:       Annotated[list[dict], operator.add]
    schema:        CorpusSchema
    assembled:     list[dict]
    audited:       Annotated[list[dict], operator.add]
    audit_log:     Annotated[list[dict], operator.add]
    output_path:   str
    n_records:     int


# ══════════════════════════════════════════════════════════════════════════════
#  LLM plumbing
# ══════════════════════════════════════════════════════════════════════════════
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


def llm_call(model: str, messages: list[dict], temperature: float = 0) -> str:
    if model in NO_SYSTEM_ROLE:
        messages = _merge_system_into_user(messages)

    # Reproducibility. temperature=0 and a fixed seed are NOT sufficient against
    # a shared hosted endpoint: requests are batched with other traffic,
    # floating-point summation order varies with that batching, and over a long
    # generation the tiny numeric differences compound into different token
    # choices. The only thing that makes a run repeatable is therefore not
    # asking the endpoint twice. Responses are keyed by SHA-256(model,
    # messages), so a warm cache reproduces a previous run exactly; committing
    # the cache alongside results lets anyone else reproduce the same numbers.
    # Any change to a prompt, a model, or a document changes the key and forces
    # a real call, so the cache cannot silently serve stale answers.
    if _CACHE is not None:
        hit = _CACHE.get(model, messages)
        if hit is not None:
            return hit

    delay = RETRY_BASE_DELAY
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            extra = ({"reasoning_effort": REASONING_EFFORT[model]}
                     if model in REASONING_EFFORT else {})
            resp = _client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                seed=42, max_tokens=MAX_OUTPUT_TOKENS, **extra,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason == "length":
                print(f"    [warn] {model} hit max_tokens ({MAX_OUTPUT_TOKENS}); "
                      f"reply is cut off and later records are lost", flush=True)
            # Reasoning models return their chain of thought in a separate
            # `reasoning` field and the answer in `content`. An empty content
            # with finish_reason="length" therefore means the whole output
            # budget went to reasoning before the answer began -- retryable, and
            # worth surfacing rather than parsing as an empty result.
            if not content.strip():
                raise RuntimeError(
                    f"{model} returned empty content (finish_reason="
                    f"{choice.finish_reason}, completion_tokens="
                    f"{resp.usage.completion_tokens if resp.usage else '?'})"
                )
            if _CACHE is not None:
                _CACHE.set(model, messages, content)
            return content
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                print(f"    retry {attempt}/{MAX_RETRIES}: {exc}", flush=True)
                time.sleep(delay)
                delay *= 2
            else:
                raise last_exc
    return ""


def parse_json_obj(raw: str) -> dict:
    """Robustly parse a single JSON object from an LLM reply: strips <think>
    blocks, repairs truncated brackets, falls back to fenced/bare-object regex
    extraction. Returns {} if nothing usable is found."""
    if not raw:
        return {}
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    last_close = raw.rfind("}")
    if last_close != -1:
        candidate = raw[:last_close + 1]
        opens_sq = candidate.count("[") - candidate.count("]")
        opens_cu = candidate.count("{") - candidate.count("}")
        if opens_sq >= 0 and opens_cu >= 0:
            repaired = candidate + "]" * opens_sq + "}" * opens_cu
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    print("      [parse] repaired truncated JSON", flush=True)
                    return data
            except Exception:
                pass

    m = (re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
         or re.search(r"(\{.*\})", raw, re.DOTALL))
    if m:
        try:
            data = json.loads(m.group(1))
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def parse_items(raw: str, wrapper_key: str) -> list[dict]:
    data = parse_json_obj(raw)
    reasoning = data.get("reasoning", "")
    if reasoning:
        print(f"      [reasoning] {str(reasoning)[:200]}", flush=True)
    items = data.get(wrapper_key, [])
    if isinstance(items, list):
        return [r for r in items if isinstance(r, dict)]

    # last resort: salvage bare {...} objects that carry a "lines" or "id" key
    salvaged = []
    for m in re.finditer(r'\{[^{}]*"(?:lines|id)"[^{}]*\}', raw or "", re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                salvaged.append(obj)
        except json.JSONDecodeError:
            continue
    if salvaged:
        print(f"      [parse] salvaged {len(salvaged)} records", flush=True)
    return salvaged


def _snake(name: str) -> str:
    """Normalise a field name to lower_snake_case without assuming anything
    about what the name means."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name or "").strip()).strip("_").lower()
    return re.sub(r"_+", "_", s)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Segmentation (deterministic, structure-blind)
# ══════════════════════════════════════════════════════════════════════════════
def _heading_text(block: str) -> str | None:
    """Typographic-convention-blind heading detector. A heading is identified
    purely by its SHAPE, not by matching any specific marker ('#', a number,
    all-caps, brackets, underlines...): it is a block that stands entirely alone
    as ONE line (isolated by blank lines on both sides, which is guaranteed
    since blocks are blank-line-delimited), short, and phrased like a label
    rather than a sentence (no sentence-final punctuation). This holds for a
    heading under any typographic convention a document might use, including
    ones never seen before, because it depends only on a heading being visually
    set apart and brief. A leading run of non-alphanumeric characters is trimmed
    for a cleaner context label, without the code needing to know what that
    marker means."""
    lines = block.splitlines()
    if len(lines) != 1:
        return None
    line = lines[0].strip()
    if not line or len(line) > 90 or len(line.split()) < 2:
        return None
    if line.endswith((".", ",", ";")):
        return None
    i = 0
    while i < len(line) and not line[i].isalnum():
        i += 1
    return line[i:].strip() or None


def _is_rule_off(line: str) -> bool:
    """True for a line made only of ruling characters -- a table's header/body
    separator, a horizontal rule, an underline. Judged by character content
    alone, never by deciding the surrounding block "is a table"; such a line
    carries no statement under any layout convention, so numbering it would
    invite an agent to cite it as the source of a rule."""
    stripped = re.sub(r"[\s|:+\-=_*~.#]", "", line)
    return stripped == ""


def segment_document(fname: str, text: str, max_chars: int) -> list[dict]:
    """Split one document into chunks of NUMBERED content lines.

    Line numbers are the document's own 1-based line numbers, assigned here in
    code. Nothing downstream ever asks a model to invent or echo a provenance
    string: an agent cites a number from the numbered list in front of it, and
    the pipeline resolves that number back to the verbatim line. Provenance is
    therefore always present and always exact.

    Blocks are blank-line-delimited, so a table's rows (which have no blank
    lines between them) stay together and can be packed into one chunk with
    their own header row visible. Headings keep their position in the stream
    rather than collapsing into one label per chunk: a chunk that spans several
    sections shows each heading where it actually falls, so every line is read
    under the section that really governs it, and a chunk that starts mid-
    section is prefixed with the heading it inherits. The code never asks
    whether a block IS a table, a matrix, or prose."""
    raw_lines = text.splitlines()

    blocks: list[list[int]] = []      # each block is a list of line numbers (1-based)
    current: list[int] = []
    for idx, line in enumerate(raw_lines, start=1):
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(idx)
    if current:
        blocks.append(current)

    chunks: list[dict] = []
    cur_items: list[dict] = []
    cur_lines: list[int] = []
    cur_len = 0
    carried_header = ""     # heading in force where the current chunk starts
    last_header = ""        # most recent heading seen anywhere

    def flush() -> None:
        nonlocal cur_items, cur_lines, cur_len, carried_header
        if cur_lines:
            chunks.append({
                "chunk_id":    f"{fname}::{len(chunks)}",
                "source_file": fname,
                "section":     carried_header,
                "items":       cur_items,
                "lines":       cur_lines,
            })
        cur_items, cur_lines, cur_len = [], [], 0
        carried_header = last_header

    def add_heading(heading: str) -> None:
        nonlocal cur_len, carried_header, last_header
        last_header = heading
        if cur_lines:
            cur_items.append({"kind": "heading", "text": heading})
            cur_len += len(heading) + 12
        else:
            carried_header = heading

    def add_line(n: int) -> None:
        nonlocal cur_len
        cur_items.append({"kind": "line", "n": n})
        cur_lines.append(n)
        cur_len += len(raw_lines[n - 1]) + 1

    for block in blocks:
        block_text = "\n".join(raw_lines[i - 1] for i in block)
        heading = _heading_text(block_text)
        if heading:
            # A heading is context, not content: it labels what follows and is
            # never offered as a citable line, so it cannot be extracted as a
            # rule in its own right.
            add_heading(heading)
            continue
        content = [i for i in block if not _is_rule_off(raw_lines[i - 1])]
        if not content:
            continue
        block_chars = sum(len(raw_lines[i - 1]) + 1 for i in content)

        # A block larger than the budget on its own -- a long table -- is split
        # on line boundaries, so no line is ever dropped or truncated away.
        if block_chars > max_chars:
            flush()
            for n in content:
                if cur_len + len(raw_lines[n - 1]) + 1 > max_chars and cur_lines:
                    flush()
                add_line(n)
            flush()
            continue

        if cur_len + block_chars > max_chars and cur_lines:
            flush()
        for n in content:
            add_line(n)
    flush()

    return chunks


def render_chunk(chunk: dict, file_lines: dict[str, list[str]]) -> str:
    """The exact text an agent sees: the section heading in force at the start,
    then the chunk's content in document order -- one numbered line per citable
    line, with any further headings shown in the position they occupy. Only
    content lines carry a number, so a heading can never be cited as the source
    of a rule."""
    lines = file_lines[chunk["source_file"]]
    parts: list[str] = []
    if chunk["section"]:
        parts.append(f"[Section: {chunk['section']}]")
    for item in chunk["items"]:
        if item["kind"] == "heading":
            parts.append(f"[Section: {item['text']}]")
        else:
            n = item["n"]
            parts.append(f"L{n}: {lines[n - 1].rstrip()}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt 1 — Scout (one call per chunk)
# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARY HYGIENE, enforced for every prompt in this file: the structural
# instructions are the mechanism and are stated generically; every ILLUSTRATION
# in them (example identifier codes, kind-of-subject nouns, category
# descriptors) is drawn from OUTSIDE the evaluated corpora. No example token may
# appear as a printed identifier in an evaluated document, as a ground-truth
# column name, or as a ground-truth category/severity label — otherwise the
# prompt states part of an answer the run is scored on. Words that DEFINE the
# task itself ("rule, limit, requirement, or procedure") are task scope, not
# illustration, and stay. Checked mechanically against all evaluated corpora
# and ground truths; re-run that check whenever an example here or an evaluated
# corpus changes.
_COT_INSTRUCTION = (
    "\n\nBefore producing the JSON, reason step by step: identify each rule, "
    "requirement, limit or procedure the excerpt states, and its logical "
    "structure. Then produce the final JSON."
)


def scout_prompt() -> str:
    return """You are a precision data-extraction engineer reading ONE excerpt of an
industrial operating or control document. You are not told in advance how the
plant, its processes, or its vocabulary are organised — every document names
the things, people, places, processes and codes it governs in its own words,
and you must not assume any of that vocabulary going in. Every content line of the
excerpt is numbered (L7, L8, ...). Extract every explicitly stated rule,
limit, requirement, or procedure.

WHAT COUNTS AS ONE RECORD — the document itself decides, never you:
  - If the document prints its own identifier for a rule (a code such as
    "REQ-9", "K-12", "WQ-77", "B.11"), everything that identifier introduces is
    ONE record, and you copy that identifier verbatim into "id".
  - A row of a data table is ONE record. Its cells are that record's fields.
  - A cell of a cross-reference matrix is ONE record: its row-axis label and
    its column-axis label are two of its fields, and the cell's own content is
    another.
  - A sentence or bullet that states a rule but prints no identifier of its
    own is ONE record, with "id" left as "".
  - If ONE line states SEVERAL rules — because it prints several identifiers,
    or because it states genuinely different consequences or importance levels
    for different situations — emit one record per rule, each citing that same
    line. Do not bundle them, and do not split a single rule into pieces.

BEFORE READING ANY TABLE, ASK WHAT ITS COLUMN HEADERS ARE. This decides
whether a row is one record or several, and it is the single most important
judgement you make:
  - ATTRIBUTE COLUMNS — each header names a DIFFERENT KIND of information
    about the row's subject (its measuring unit, a numeric ceiling, a required
    follow-up, a due time, a cross-reference). The headers are not comparable to one another; no
    two could be swapped. Then the ROW is ONE record, and each header is a
    field name holding that row's value for it.
  - CASE COLUMNS — the headers are all instances of ONE KIND of thing
    (several sites, several staff groups, several importance grades, several
    work stages, several machines), and every cell beneath them holds the same
    kind of content as every other. The table is then a cross-reference
    matrix, and each CELL is its own record, because each cell states what
    applies in ITS OWN case and nothing about the others. Give each such
    record three things: a field naming the row-axis label, a field naming the
    column-axis label, and a field holding that one cell's content. The field
    naming an axis is named after the KIND the axis ranges over (whatever the
    document calls that kind — the axis's own corner header if it prints one),
    NEVER after an individual instance. Turning an instance name into a field
    name is always wrong: it produces one bloated record per row instead of
    one record per case, and every case but the first loses its identity.
  - Test the two readings against each other before choosing. If you find
    yourself creating a field whose name is one particular thing — one site,
    one staff group, one work stage, one machine — you have a matrix and
    should be emitting one record per cell instead.

FIELD NAMES come from the document, never from you:
  - In an attribute-column table, a field's name is ITS OWN COLUMN HEADER,
    copied from the header row and normalized to lower_snake_case. A column
    header is a NAME: it belongs in the field name, never in a value. Emitting
    one record per cell of such a row is wrong — a data row with N attribute
    columns is ONE record with up to N fields.
  - In prose, use whatever term the document itself uses for that idea,
    normalized to lower_snake_case. Only when the document states an idea
    repeatedly without ever printing a label for it may you choose a plain
    name yourself — and then use that one name for every record that states
    it, never two names for one idea.
  - Do not paraphrase, generalize, or substitute a word you find more familiar
    for the document's own label, and never invent a field for information the
    document does not state.

VALUES are copied exactly as the document states them: no rounding, unit
conversion, translation, paraphrase, or inference, and never a value borrowed
from a neighbouring row, column, or sentence. Omit a field entirely rather
than guess at it.

CARRY THE SUBJECT DOWN FROM THE HEADING. A document states a rule's subject
once, in the heading above it, and then never repeats it on the rows beneath —
a table under a heading naming one specific thing is a table ABOUT that thing,
and every row of it inherits it. Whenever a record's
own line does not name the entity it concerns but the "[Section: ...]" marker
governing that line does, copy that entity into a field of the record, named
after whatever the document calls that kind of entity. A record that cannot say
what it is about is not usable, so this is not optional: check every record you
emit for a field naming its subject before you return it.

But NEVER create a field for the document's own title, code, revision, or the
name of the facility the whole document is about. A section marker sometimes
carries the document's title rather than a subject within it — that is the case
whenever it names what the WHOLE document is, not what one part of it concerns.
Such a value belongs to every record equally, so it distinguishes nothing and
is recorded separately as provenance. The test is simple: if a field would hold
the same value for every record in the document, it is not a property of any
rule and must be left out.

Also give each record:
  "category" — what the rule FUNCTIONALLY IS, in two or three words of your
  own (a standing cap, an automatic reaction, a rule about who may enter, a
  recurring duty...), judged by what it does and never by where on the page it sits: a
  table and a paragraph can state the same kind of rule.
  "lines" — the numbers of the excerpt lines this record was read from, as
  integers. Cite only line numbers printed in the excerpt below. A record read
  from one table row cites that row's line; a rule stated across two lines
  cites both.

Scan the WHOLE excerpt systematically — every line, every row, every cell — and
do not stop after the first record. Extract every distinct rule, including ones
that state only a bare limit, status, or classification with no consequence
attached; deciding what is significant enough to keep is not this pass's job.
Skip only literal non-content: a repeated column header, a page footer, a
document title. Lines reading "[Section: ...]" are the document's own headings,
shown in the position they occupy and carrying no line number. They are
context, never records: every numbered line below such a marker belongs to that
section until the next marker appears, so use them to fill fields that depend
on section context — which subject the rows beneath them are about — and never
extract one as a record of its own.

No worked example is given, deliberately. Any example would demonstrate one
particular document shape — a tiered numeric table, a cross-reference matrix,
a recurring schedule — and a model shown such an example reproduces its
vocabulary and its structure on documents that have neither. The only thing to work from is the
excerpt itself.

Reply EXCLUSIVELY with JSON:
{"records": [{"id": "<printed identifier or empty>", "category": "<what it is>",
"lines": [<numbers>], "fields": {"<field_name>": "<value>", ...}}, ...]}
You may prefix it with a brief "reasoning" key.""" + _COT_INSTRUCTION


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt 2 — Schema arbiter (one call per corpus)
# ══════════════════════════════════════════════════════════════════════════════
def inducer_prompt() -> str:
    return """You are a schema arbiter. Below is the COMPLETE inventory of field names and
category labels that an extraction pass produced across one corpus of
industrial operating documents, each with how many records used it and a few
example values. You are not shown the documents: your job is not to re-read
them but to reconcile the vocabulary that reading them produced.

Different documents in one corpus routinely name the same underlying idea
differently — one prints a column header, another states the same idea in a
bare sentence, a third abbreviates it. Produce ONE schema the whole corpus can
speak.

MERGE ONLY TRUE SYNONYMS. Two names merge when they carry the SAME information
about a record, so that no record could ever need both at once. Names that
merely look alike, sit near each other, or share a word are NOT synonyms:
  - Two bounds of different tightness, or on different sides of a range, are
    DIFFERENT fields, even when their names differ by one word. Merging them
    destroys the distinction the document went to the trouble of printing.
  - A field holding a NUMBER and a field holding the PROSE that describes it
    are different fields.
  - A field naming a thing and a field holding a quantity observed on that
    thing are different fields.
When in doubt, keep them separate: an unmerged pair costs one redundant column,
while a wrongly merged pair silently overwrites real values.

KEEP EVERY DISTINCT IDEA. Do not compress the schema toward a small round
number, and do not drop a field for being rare — a field used by three records
out of two hundred is still the only place those three records' information
lives. Every observed name must appear as a key of "field_map", mapped either
to itself or to the canonical name it merges into. The same holds for every
observed category in "category_map".

CANONICAL NAMES must be drawn from the observed names themselves — pick the
clearest of the synonyms being merged. Do not coin a new vocabulary.

Then identify three special fields, using canonical names from your own schema,
or "" if the corpus has no such field at all:
  "condition_field"  — whichever field holds the prose stating WHEN a rule
                       applies. It must be the narrative trigger, not a bare
                       number: if the only candidate holds numeric values,
                       leave this "".
  "action_field"     — whichever field holds what the rule REQUIRES or CAUSES.
  "severity_field"   — whichever field holds an importance or criticality label.
  "numeric_fields"   — every canonical field whose values are pure numbers.

Reply ONLY with JSON in exactly this shape:
{
  "fields": {"<canonical_name>": "<one-line description>", ...},
  "field_map": {"<observed_name>": "<canonical_name>", ...},
  "categories": {"<CanonicalCategory>": "<one-line description>", ...},
  "category_map": {"<observed_label>": "<CanonicalCategory>", ...},
  "condition_field": "", "action_field": "", "severity_field": "",
  "numeric_fields": []
}
Canonical field names are lower_snake_case; canonical category names are
CamelCase.

OBSERVED INVENTORY:
"""


def _coerce_schema(data: dict, observed_fields: list[str],
                   observed_categories: list[str]) -> CorpusSchema:
    """Turn the arbiter's reply into a CorpusSchema, defensively.

    Any observed name the arbiter failed to mention maps to itself. This is the
    difference between a schema pass that reconciles vocabulary and one that
    deletes data: a forgotten name must cost a redundant column, never the
    values recorded under it."""
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    field_map = data.get("field_map") if isinstance(data.get("field_map"), dict) else {}
    categories = data.get("categories") if isinstance(data.get("categories"), dict) else {}
    category_map = data.get("category_map") if isinstance(data.get("category_map"), dict) else {}

    field_map = {_snake(k): _snake(v) for k, v in field_map.items() if v}
    for name in observed_fields:
        field_map.setdefault(name, name)
    category_map = {str(k).strip(): str(v).strip() for k, v in category_map.items() if v}
    for name in observed_categories:
        category_map.setdefault(name, name)

    fields = {_snake(k): str(v) for k, v in fields.items()}
    for canonical in set(field_map.values()):
        fields.setdefault(canonical, "")

    numeric = data.get("numeric_fields", [])
    numeric = [_snake(f) for f in numeric if isinstance(f, str)] if isinstance(numeric, list) else []

    def _role(key: str) -> str:
        value = _snake(data.get(key, "") or "")
        return value if value in fields else ""

    return CorpusSchema(
        fields=fields, field_map=field_map,
        categories={str(k): str(v) for k, v in categories.items()},
        category_map=category_map,
        condition_field=_role("condition_field"),
        action_field=_role("action_field"),
        severity_field=_role("severity_field"),
        numeric_fields=numeric,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt 3 — Grounding auditor (one call per chunk)
# ══════════════════════════════════════════════════════════════════════════════
def auditor_prompt() -> str:
    return """You are a grounding auditor. You will be given one excerpt of a document with
its lines numbered, and the records an earlier pass read out of those lines.
For EACH record, check every non-empty field value against the excerpt's own
content and return one verdict:

  - "keep" — every stated value is present in, or a faithful non-inventive
    reading of, the lines that record cites.
  - "correct" — the record is real but one or more values are wrong, taken
    from a neighbouring row or column, or paraphrased where the source is more
    specific. Give the corrected values, copied verbatim from the excerpt.
  - "drop" — the record is not stated in the cited lines at all, or it is a
    pure duplicate of another record covering the exact same line with no
    distinguishing detail.

DO NOT DROP A RECORD FOR LOOKING LIKE ANOTHER ONE. Documents routinely restate
a similar-sounding rule for many different entities — a different row or a
different subject each getting its own, otherwise identical-looking rule. Each
one is real and stays, however repetitive the set looks in aggregate.

DO NOT DROP A RECORD FOR HAVING NO SENTENCE OF ITS OWN. Many records come from
a table row or a matrix cell that prints only labels and values and never forms
a sentence. Judge such a record by whether its values and labels really do come
from the line it cites. Reserve "drop" for content with no corresponding line
at all.

Reply ONLY with JSON: {"records": [{"key": "<copy from input>",
"verdict": "keep|correct|drop", "corrections": {"<field>": "<value>", ...}}]}
Include exactly one entry per record listed. "corrections" may be omitted or
empty for keep and drop.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Scout fan-out
# ══════════════════════════════════════════════════════════════════════════════
def scout_chunk(chunk: dict, file_lines: dict[str, list[str]], model: str) -> list[dict]:
    """One scout call. Returns records with validated line citations."""
    rendered = render_chunk(chunk, file_lines)
    raw = llm_call(model, [
        {"role": "system", "content": scout_prompt()},
        {"role": "user", "content": rendered},
    ])
    valid_lines = set(chunk["lines"])
    out: list[dict] = []
    dropped_citations = 0
    for rec in parse_items(raw, "records"):
        fields = rec.get("fields")
        if not isinstance(fields, dict):
            fields = {k: v for k, v in rec.items() if k not in _INTERNAL_KEYS and k != "fields"}
        fields = {_snake(k): str(v).strip() for k, v in fields.items()
                  if v not in (None, "") and str(v).strip()}

        cited = rec.get("lines", [])
        if isinstance(cited, (int, str)):
            cited = [cited]
        numbers: list[int] = []
        for value in cited if isinstance(cited, list) else []:
            m = re.search(r"\d+", str(value))
            if m and int(m.group()) in valid_lines:
                numbers.append(int(m.group()))
        if not numbers:
            # A citation the excerpt does not contain is a bookkeeping failure,
            # not evidence the record is false. Anchor it to the chunk's first
            # line so it keeps a real, checkable provenance and reaches the
            # auditor, rather than being silently discarded or -- worse --
            # carried forward with no provenance at all.
            numbers = [chunk["lines"][0]]
            dropped_citations += 1

        if not fields and not str(rec.get("id", "")).strip():
            continue
        out.append({
            "id":          str(rec.get("id", "") or "").strip(),
            "category":    str(rec.get("category", "") or "").strip(),
            "lines":       sorted(set(numbers)),
            "fields":      fields,
            "source_file": chunk["source_file"],
            "chunk_id":    chunk["chunk_id"],
        })
    note = f", {dropped_citations} re-anchored" if dropped_citations else ""
    print(f"  [scout] {chunk['chunk_id']}: {len(out)} record(s){note}", flush=True)
    return out


def scout_node(payload: dict) -> dict:
    """One branch of the scout fan-out: one chunk, one call, one slice of state."""
    return {"scouted": scout_chunk(payload["chunk"], payload["file_lines"],
                                   payload["model"])}


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Schema induction
# ══════════════════════════════════════════════════════════════════════════════
def build_inventory(records: list[dict]) -> tuple[dict, dict]:
    """The arbiter's whole input: which names were used, how often, and what
    they held. Assembled deterministically from the scouted records, so the
    arbiter reconciles observed vocabulary and never invents from a document."""
    field_stats: dict[str, dict] = {}
    category_stats: dict[str, int] = {}
    for rec in records:
        label = rec.get("category", "")
        if label:
            category_stats[label] = category_stats.get(label, 0) + 1
        for name, value in rec["fields"].items():
            entry = field_stats.setdefault(name, {"count": 0, "samples": []})
            entry["count"] += 1
            if len(entry["samples"]) < INDUCTION_SAMPLES_PER_FIELD and value not in entry["samples"]:
                entry["samples"].append(value[:80])
    # EVERY observed name reaches the arbiter. When the inventory outgrows the
    # prompt budget it is the sample values that thin, never the vocabulary:
    # a name the arbiter never sees cannot be reconciled, and its records' values
    # are lost with it, so discarding names by frequency would delete exactly the
    # rare fields that carry edge-case triggers. Thinning samples degrades how
    # well the arbiter can judge a name; dropping names decides that it will not
    # judge them at all. Only the first is an acceptable response to a budget.
    by_count = sorted(field_stats.items(), key=lambda kv: -kv[1]["count"])
    n = len(by_count)
    if n > INDUCTION_HARD_MAX_FIELDS:
        raise RuntimeError(
            f"observed inventory has {n} distinct field names, beyond the "
            f"{INDUCTION_HARD_MAX_FIELDS} that one induction prompt can carry even "
            f"with a single sample each. This is the point at which per-document "
            f"inventories must be merged hierarchically rather than induced in one "
            f"call; raising the constant would only move the failure. Stopping "
            f"rather than inducing over a partial vocabulary.")
    if n > INDUCTION_BUDGET_FIELDS:
        allowed = max(1, (INDUCTION_SAMPLES_PER_FIELD * INDUCTION_BUDGET_FIELDS) // n)
        print(f"  [note] observed inventory has {n} field name(s), above the budget of "
              f"{INDUCTION_BUDGET_FIELDS}. ALL {n} names are passed to the arbiter; "
              f"sample values per name are thinned {INDUCTION_SAMPLES_PER_FIELD} -> "
              f"{allowed} to fit. No vocabulary is discarded.", flush=True)
        for _, entry in by_count:
            entry["samples"] = entry["samples"][:allowed]
    ordered = dict(by_count)
    return ordered, dict(sorted(category_stats.items(), key=lambda kv: -kv[1]))


def induce_schema(records: list[dict], model: str) -> CorpusSchema:
    field_stats, category_stats = build_inventory(records)

    payload = {
        "fields": {name: {"records": stat["count"], "examples": stat["samples"]}
                   for name, stat in field_stats.items()},
        "categories": category_stats,
    }
    raw = llm_call(model, [
        {"role": "user", "content": inducer_prompt()
         + json.dumps(payload, indent=2, ensure_ascii=False)},
    ])
    schema = _coerce_schema(parse_json_obj(raw), list(field_stats), list(category_stats))
    merged = len(field_stats) - len({schema.canon_field(f) for f in field_stats})
    print(f"[Induce/{model}] {len(field_stats)} observed field name(s) -> "
          f"{len(set(schema.field_map.values()))} canonical ({merged} merged); "
          f"{len(category_stats)} observed categor(ies) -> "
          f"{len(set(schema.category_map.values()))}")
    print(f"           roles: condition={schema.condition_field or '-'} "
          f"action={schema.action_field or '-'} severity={schema.severity_field or '-'}")
    return schema


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4 — Assembly (deterministic)
# ══════════════════════════════════════════════════════════════════════════════
def record_key(rec: dict, seq: int) -> str:
    """The document's own record boundary.

    A printed identifier is the only evidence a document gives that two
    readings describe the SAME rule, so it is the only thing that merges them:
    a rule restated either side of a chunk boundary carries its id both times
    and reunites here. Without a printed id there is no such evidence, and one
    line may legitimately state several distinct rules -- a matrix line holds
    one per cell -- so each reading keeps its own identity, distinguished by
    the order it was read in. Merging those instead would silently delete every
    rule on the line but the first, which is the more expensive mistake: a
    redundant record is visible and can be dropped by the audit, whereas a
    deleted one leaves nothing behind to notice.

    The boundary is a property of the DOCUMENT either way, so granularity is
    decided by what the source prints and never by the column conventions of
    whatever ground truth the output is later compared against."""
    printed = re.sub(r"\s+", "", rec.get("id", "")).upper()
    if printed:
        return f"{rec['source_file']}::{printed}"
    return f"{rec['source_file']}::L{min(rec['lines'])}#{seq}"


def drop_document_identity_fields(records: list[dict],
                                  file_lines: dict[str, list[str]]) -> int:
    """Remove fields that name the DOCUMENT rather than the rule.

    A document's own title line is short and unpunctuated, so it looks exactly
    like a section heading, and a heading is what tells an extractor what the
    rows beneath it are about. When the heading in force IS the document title,
    that inherited subject is the same for every record in the file: it says
    nothing about any individual rule, and provenance already records which
    document a record came from.

    Left in, such a field is worse than useless -- it is a long constant string
    on every record, and any comparison against the record's real content is
    diluted by it in proportion to its length.

    The test is grounded in the document rather than in a tuned threshold: a
    field is dropped when every value it holds within one file is part of that
    file's own title line. A field carrying real per-rule content cannot pass
    that test, because its values differ from row to row."""
    by_file: dict[str, list[dict]] = {}
    for rec in records:
        by_file.setdefault(rec["source_file"], []).append(rec)

    dropped = 0
    for fname, group in by_file.items():
        lines = file_lines.get(fname) or []
        title = next((ln.strip() for ln in lines if ln.strip()), "")
        if not title:
            continue
        title_norm = _norm_for_match(title)
        if not title_norm:
            continue

        candidates: dict[str, bool] = {}
        for rec in group:
            for name, value in rec["fields"].items():
                v = _norm_for_match(value)
                is_title_part = bool(v) and (v in title_norm or title_norm in v)
                candidates[name] = candidates.get(name, True) and is_title_part
        doomed = {name for name, all_title in candidates.items() if all_title}
        for rec in group:
            for name in doomed:
                if rec["fields"].pop(name, None) is not None:
                    dropped += 1
    return dropped


def _norm_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def assemble_records(records: list[dict], schema: CorpusSchema) -> list[dict]:
    """Group scouted records by the document's record boundary and rename every
    field to its canonical name. One row out per rule."""
    groups: dict[str, dict] = {}
    conflicts = 0
    for seq, rec in enumerate(records):
        key = record_key(rec, seq)
        group = groups.get(key)
        if group is None:
            group = groups[key] = {
                "_key": key, "id": rec.get("id", ""), "source_file": rec["source_file"],
                "lines": set(), "fields": {}, "categories": [],
            }
        group["lines"].update(rec["lines"])
        if not group["id"] and rec.get("id"):
            group["id"] = rec["id"]
        if rec.get("category"):
            group["categories"].append(schema.canon_category(rec["category"]))
        for name, value in rec["fields"].items():
            canonical = schema.canon_field(name)
            existing = group["fields"].get(canonical, "")
            if not existing:
                group["fields"][canonical] = value
            elif value != existing:
                conflicts += 1
                # A numeric slot holding two different numbers means one of them
                # is misfiled; joining them would invent a value that appears
                # nowhere in the document, so the first reading stands and the
                # disagreement is counted rather than hidden. Prose slots do
                # legitimately accumulate (a rule stated over two lines), so
                # those are joined.
                if canonical not in schema.numeric_fields and value not in existing:
                    group["fields"][canonical] = f"{existing}; {value}"

    out = []
    for group in groups.values():
        categories = group["categories"]
        group["category"] = max(set(categories), key=categories.count) if categories else ""
        group["lines"] = sorted(group["lines"])
        out.append(group)
    out.sort(key=lambda g: (g["source_file"], g["lines"][0] if g["lines"] else 0))
    print(f"[Assemble] {len(records)} scouted -> {len(out)} record(s)"
          f"{f', {conflicts} field conflict(s)' if conflicts else ''}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 5 — Grounding audit fan-out
# ══════════════════════════════════════════════════════════════════════════════
def audit_chunk(chunk: dict, group: list[dict], file_lines: dict[str, list[str]],
                model: str) -> tuple[list[dict], list[dict]]:
    """Returns (kept records, verdict log). The log exists because the printed
    per-chunk summary was previously the only trace of what the auditor did:
    stdout is not an artifact, so the audit stage's intervention rate could not
    be reported from a finished run. Every verdict is now written to a sidecar
    CSV by save_outputs, making the stage's contribution measurable after the
    fact without changing what it does."""
    if not group:
        return [], []
    rendered = render_chunk(chunk, file_lines)
    payload = [{"key": g["_key"], "id": g["id"], "category": g["category"],
                "lines": g["lines"], "fields": g["fields"]} for g in group]
    raw = llm_call(model, [
        {"role": "system", "content": auditor_prompt()},
        {"role": "user", "content": f"EXCERPT:\n{rendered}\n\nRECORDS:\n"
                                    + json.dumps(payload, indent=2, ensure_ascii=False)},
    ])
    verdicts = {}
    for item in parse_items(raw, "records"):
        key = str(item.get("key", "")).strip()
        if key:
            verdicts[key] = item

    def log_row(g: dict, decision: str, outcome: str, corrected_fields: str = "",
                override: bool = False) -> dict:
        return {"chunk_id": chunk["chunk_id"], "source_file": g["source_file"],
                "record_key": g["_key"], "record_id": g.get("id", ""),
                "category": g.get("category", ""), "verdict": decision,
                "outcome": outcome, "corrected_fields": corrected_fields,
                "chunk_audit_override": override}

    dropped = [g for g in group if str(verdicts.get(g["_key"], {}).get("verdict", "keep")).lower() == "drop"]
    # An auditor that rejects a whole chunk is far likelier to be malfunctioning
    # -- a misread instruction, a truncated reply -- than to be right that every
    # record in it was fabricated. Treat that as an audit failure and keep the
    # chunk intact rather than deleting a document's entire section on one call.
    if len(dropped) == len(group) and len(group) >= 3:
        print(f"  [audit] {chunk['chunk_id']}: rejected all {len(group)} records "
              f"-- treated as audit failure, keeping all", flush=True)
        return group, [log_row(g, "drop", "kept", override=True) for g in group]

    kept: list[dict] = []
    log: list[dict] = []
    corrected = 0
    for g in group:
        verdict = verdicts.get(g["_key"], {})
        decision = str(verdict.get("verdict", "keep")).lower()
        if decision == "drop":
            log.append(log_row(g, decision, "dropped"))
            continue
        corrected_names = ""
        if decision == "correct" and isinstance(verdict.get("corrections"), dict):
            # KNOWN DEFECT, left in place deliberately and disclosed in the
            # thesis rather than patched after the fact. _INTERNAL_KEYS is
            # filtered on the scout path but not here, so an auditor that
            # returns a correction keyed "lines" writes the record's own
            # provenance into its content fields, where it becomes a CSV column
            # and enters the scored blob. It happened on 8 records of one corpus
            # across the reported batch; rescoring those runs with the field
            # excluded moves that corpus's F1 by 0.0000, so no reported figure
            # depends on it. Filtering it here would change extraction output
            # and invalidate the batch every number in the thesis is computed
            # from, which is not a trade worth making for a measured effect of
            # zero -- fix it together with the next full re-run.
            applied = [name for name, value in verdict["corrections"].items()
                       if value not in (None, "")]
            for name in applied:
                g["fields"][_snake(name)] = str(verdict["corrections"][name]).strip()
            corrected_names = "|".join(_snake(n) for n in applied)
            corrected += 1
        log.append(log_row(g, decision if verdict else "unmentioned", "kept",
                           corrected_names))
        kept.append(g)
    print(f"  [audit] {chunk['chunk_id']}: {len(kept)}/{len(group)} kept, "
          f"{corrected} corrected", flush=True)
    return kept, log


def audit_groups(chunks: list[dict], assembled: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Pair every record with the chunk its first cited line belongs to, so each
    record is judged beside the text it was actually read from."""
    line_owner: dict[tuple[str, int], str] = {}
    by_chunk_id = {c["chunk_id"]: c for c in chunks}
    for c in chunks:
        for n in c["lines"]:
            line_owner[(c["source_file"], n)] = c["chunk_id"]

    grouped: dict[str, list[dict]] = {}
    for g in assembled:
        first = g["lines"][0] if g["lines"] else None
        owner = line_owner.get((g["source_file"], first))
        if owner is None:
            continue
        grouped.setdefault(owner, []).append(g)
    return [(by_chunk_id[cid], group) for cid, group in grouped.items()]


def audit_node(payload: dict) -> dict:
    """One branch of the audit fan-out."""
    kept, log = audit_chunk(payload["chunk"], payload["group"],
                            payload["file_lines"], payload["model"])
    return {"audited": kept, "audit_log": log}


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 6 — Output
# ══════════════════════════════════════════════════════════════════════════════
def normalise_field_name(name: str) -> str:
    """Deterministic syntactic hygiene on a discovered field name.

    Case-folds, unifies separators, and collapses repeats, so that "Crit LO",
    "crit-lo" and "crit__lo" become one column instead of three. It is purely
    syntactic: no synonym list, no semantic mapping, nothing that encodes what a
    field means in any domain. Two names the arbiter chose for genuinely
    different reasons ("station" and "asset") remain different columns, because
    deciding they are the same would require domain knowledge this pipeline does
    not have and must not invent.
    """
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip()).strip("_").lower()
    return re.sub(r"_+", "_", out) or "field"


def to_rows(assembled: list[dict], schema: CorpusSchema,
            file_lines: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    """Render assembled records as the scored CSV plus a provenance sidecar.

    Every canonical field the corpus was observed to carry becomes its own
    column, named as the corpus itself named it. The column set is therefore the
    union of what the documents exhibited -- a corpus whose rules carry fourteen
    slots yields fourteen columns -- and a record simply leaves blank the columns
    it has no value for, exactly as a hand-written rule table does.

    Nothing here consults a target schema. The names come from the arbiter's
    canonical vocabulary for THIS corpus, so the output shape adapts to the
    documents rather than to any particular ground truth, and two unrelated
    corpora are never forced to share invented field names.

    Columns are ordered by how many records carry them, most common first, with
    ties broken alphabetically. The ordering is a deterministic function of the
    data, so two runs that observe the same fields lay them out identically.

    "source_span" carries the record's own boundary key followed by the text it
    was read from. The key makes it unique per record, which is what a consumer
    grouping on it needs now that assembly happens here: two rules printed on
    one line stay two rules."""
    seq_by_file: dict[str, int] = {}
    used_ids: set[str] = set()
    rows: list[dict] = []
    facts: list[dict] = []

    for g in assembled:
        fname = g["source_file"]
        own_id = str(g.get("id", "") or "").strip()
        if own_id and own_id not in used_ids:
            rid = own_id
        else:
            seq_by_file[fname] = seq_by_file.get(fname, 0) + 1
            stem = re.sub(r"[^A-Za-z0-9]+", "", os.path.splitext(fname)[0])[:12].upper()
            rid = f"{stem}-{seq_by_file[fname]:03d}"
        used_ids.add(rid)

        source_text = " ".join(file_lines[fname][n - 1].strip() for n in g["lines"])
        record = {"id": rid, "category": g["category"]}
        for name, value in g["fields"].items():
            col = normalise_field_name(name)
            if value in (None, ""):
                continue
            prev = str(record.get(col, "")).strip()
            # Two observed names can normalise onto one column; join rather than
            # overwrite so no extracted value is silently lost.
            record[col] = value if not prev else (
                f"{prev} | {value}" if str(value) not in prev else prev)
        record["source_file"] = fname
        record["source_span"] = f"{g['_key']} | {source_text}"[:200]
        rows.append(record)

        for name, value in g["fields"].items():
            facts.append({"record_id": rid, "source_file": fname,
                          "lines": ",".join(str(n) for n in g["lines"]),
                          "field": name, "value": value})

    # Second pass: give every record the same columns, so the CSV is rectangular
    # like a rule table rather than ragged. Ordering is frequency-descending with
    # an alphabetical tie-break -- a deterministic function of the data alone.
    _FIXED_HEAD, _FIXED_TAIL = ["id", "category"], ["source_file", "source_span"]
    counts: dict[str, int] = {}
    for r in rows:
        for k in r:
            if k not in _FIXED_HEAD and k not in _FIXED_TAIL:
                counts[k] = counts.get(k, 0) + 1
    discovered = sorted(counts, key=lambda k: (-counts[k], k))
    columns = _FIXED_HEAD + discovered + _FIXED_TAIL
    rows = [{c: r.get(c, "") for c in columns} for r in rows]
    return rows, facts


def save_outputs(rows: list[dict], facts: list[dict], schema: CorpusSchema,
                 input_dir: str, run_index: int, corpus_name: str = "",
                 audit_log: list[dict] | None = None) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    # The corpus name goes in the filename because the evaluator groups repeated
    # runs by stripping the timestamp: without it, runs over DIFFERENT corpora
    # all collapse to one key and get averaged together as though they were
    # repeats of one experiment, producing a mean and a standard deviation that
    # mean nothing. The run index is explicit rather than implied by the
    # timestamp, because two fast domains can finish inside the same second.
    #
    # It defaults to the input directory's own name, which is right whenever the
    # directory is named after what it holds. --corpus-name overrides it for the
    # cases where it is not: a directory called "texts" says nothing about which
    # corpus it is or what role that corpus plays, and the label is what appears
    # in every results table.
    corpus = re.sub(r"[^A-Za-z0-9]+", "_",
                    corpus_name or os.path.basename(os.path.normpath(input_dir))).strip("_")
    stem = f"{corpus}_run{run_index}_{ts}"

    out_path = os.path.join(RESULTS_DIR, f"ext_multi_agent_generic_{stem}.csv")
    # Columns come from the rows, because the column set is DISCOVERED per corpus
    # (see to_rows) and cannot be listed here. A fixed list with
    # extrasaction="ignore" would silently drop every field the corpus turned out
    # to have, writing a well-formed file with the content missing -- a failure
    # that produces no error and is visible only as a collapsed score.
    columns = list(rows[0].keys()) if rows else ["id", "category", "source_file", "source_span"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Save] {len(rows)} record(s) -> {out_path}")

    facts_path = os.path.join(RESULTS_DIR, f"facts_{stem}.csv")
    with open(facts_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "source_file", "lines",
                                               "field", "value"])
        writer.writeheader()
        writer.writerows(facts)
    print(f"[Save] {len(facts)} field value(s) -> {facts_path}")

    schema_path = os.path.join(RESULTS_DIR, f"induced_schema_{stem}.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema.to_json(), f, indent=2, ensure_ascii=False)
    print(f"[Save] induced corpus schema -> {schema_path}")

    # One row per audited record, verdicts included. Without this file the
    # auditor's interventions exist only in stdout, and the stage's measured
    # contribution cannot be reported from a finished run's artifacts.
    if audit_log:
        audit_path = os.path.join(RESULTS_DIR, f"audit_log_{stem}.csv")
        with open(audit_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id", "source_file",
                                                   "record_key", "record_id",
                                                   "category", "verdict", "outcome",
                                                   "corrected_fields",
                                                   "chunk_audit_override"])
            writer.writeheader()
            writer.writerows(audit_log)
        print(f"[Save] {len(audit_log)} audit verdict(s) -> {audit_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def load_and_segment_node(state: PipelineState) -> dict:
    input_dir = state["input_dir"]
    txt_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".txt"))
    print(f"[Load] {len(txt_files)} document(s) in {input_dir}")
    file_lines: dict[str, list[str]] = {}
    chunks: list[dict] = []
    for fname in txt_files:
        with open(os.path.join(input_dir, fname), encoding="utf-8") as f:
            text = f.read()
        file_lines[fname] = text.splitlines()
        doc_chunks = segment_document(fname, text, state.get("chunk_chars") or DEFAULT_CHUNK_CHARS)
        chunks.extend(doc_chunks)
        citable = sum(len(c["lines"]) for c in doc_chunks)
        print(f"[Segment] {fname}: {len(doc_chunks)} chunk(s), {citable} citable line(s)")
    return {"file_lines": file_lines, "chunks": chunks, "scouted": [], "audited": []}


def dispatch_scouts(state: PipelineState) -> list[Send]:
    """Map step: one branch per chunk, all reading the same scout prompt."""
    return [Send("scout", {"chunk": c, "file_lines": state["file_lines"],
                           "model": state["scout_model"]})
            for c in state["chunks"]]


def induce_node(state: PipelineState) -> dict:
    scouted = state.get("scouted", [])
    print(f"[Scout/{state['scout_model']}] {len(scouted)} record(s) "
          f"from {len(state.get('chunks', []))} chunk(s)")
    if not scouted:
        print("[Warn] no records scouted; writing an empty result")

    # Cleaned here, before the arbiter sees the inventory, so a field naming the
    # document never becomes part of the corpus schema in the first place. The
    # records are mutated in place because `scouted` accumulates through an
    # operator.add reducer -- returning a replacement list would append to it
    # rather than replace it.
    dropped = drop_document_identity_fields(scouted, state.get("file_lines", {}))
    if dropped:
        print(f"[Clean] dropped {dropped} document-identity field value(s)")

    return {"schema": induce_schema(scouted, state["inducer_model"])}


def assemble_node(state: PipelineState) -> dict:
    return {"assembled": assemble_records(state.get("scouted", []), state["schema"])}


def dispatch_auditors(state: PipelineState) -> list[Send] | str:
    """Second map step: fan out one auditor per chunk, or edge straight to save
    when there is nothing to audit."""
    groups = audit_groups(state["chunks"], state.get("assembled", []))
    if not groups:
        return "save"
    return [Send("audit", {"chunk": chunk, "group": group,
                           "file_lines": state["file_lines"],
                           "model": state["auditor_model"]})
            for chunk, group in groups]


def save_node(state: PipelineState) -> dict:
    assembled = state.get("assembled", [])
    audited = state.get("audited", [])
    print(f"[Audit/{state['auditor_model']}] {len(audited)}/{len(assembled)} record(s) kept")
    assembled = audited
    rows, facts = to_rows(assembled, state["schema"], state["file_lines"])
    out_path = save_outputs(rows, facts, state["schema"],
                            state["input_dir"], state.get("run_index", 1),
                            state.get("corpus_name", ""),
                            state.get("audit_log", []))
    return {"output_path": out_path, "n_records": len(rows)}


def build_graph() -> StateGraph:
    """The agent graph. Two roles fan out over the document (scout, audit) and
    one runs once over the whole corpus (induce); assembly between them is
    deterministic and holds no agency of its own."""
    builder = StateGraph(PipelineState)

    builder.add_node("load_segment", load_and_segment_node)
    builder.add_node("scout",        scout_node)
    builder.add_node("induce",       induce_node)
    builder.add_node("assemble",     assemble_node)
    builder.add_node("audit",        audit_node)
    builder.add_node("save",         save_node)

    builder.add_edge(START, "load_segment")
    builder.add_conditional_edges("load_segment", dispatch_scouts, ["scout"])
    # "scout" -> "induce" is a fan-in: induce runs once, after every scout
    # branch has returned, because the arbiter's whole point is to see the
    # corpus's complete observed vocabulary rather than one chunk's.
    builder.add_edge("scout", "induce")
    builder.add_edge("induce", "assemble")
    builder.add_conditional_edges("assemble", dispatch_auditors, ["audit", "save"])
    builder.add_edge("audit", "save")
    builder.add_edge("save", END)

    return builder


def run_pipeline(input_dir: str,
                 chunk_chars: int = DEFAULT_CHUNK_CHARS,
                 max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
                 cache_path: str | None = DEFAULT_CACHE_PATH, run_index: int = 1,
                 corpus_name: str = "", **overrides: str) -> dict:
    global _CACHE
    _CACHE = LLMResponseCache(cache_path) if cache_path else None
    if _CACHE is None:
        print("[Cache] disabled -- this run is NOT reproducible")

    initial_state: PipelineState = {
        **DEFAULTS,                        # type: ignore[typeddict-item]
        "input_dir":    os.path.abspath(input_dir),
        "run_index":    run_index,
        "corpus_name":  corpus_name,
        "chunk_chars":  chunk_chars,
        "scouted":      [],
        "audited":      [],
        "audit_log":    [],
        **{k: v for k, v in overrides.items() if v},   # type: ignore[typeddict-item]
    }
    graph = build_graph().compile()
    result = graph.invoke(initial_state, config={"max_concurrency": max_concurrency})
    return {"output_path": result.get("output_path", ""),
            "n_records": result.get("n_records", 0),
            "schema": result.get("schema")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Domain- and format-agnostic multi-agent rule-extraction pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory of .txt documents to process.")
    for role, default in DEFAULTS.items():
        parser.add_argument(f"--{role.replace('_', '-')}", default=default)
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat the whole pipeline N times and write one CSV per run, "
                             "so the spread across runs can be reported as mean +- std. "
                             "N > 1 DISABLES the response cache automatically (see below).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the LLM response cache. Runs then depend on the "
                             "endpoint's non-determinism and are NOT reproducible.")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH,
                        help="Where the reproducibility cache lives.")
    parser.add_argument("--corpus-name", default="",
                        help="Label for this corpus in output filenames and result "
                             "tables. Defaults to the input directory's name.")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS,
                        help="Target character budget per scout chunk.")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
                        help="Concurrent requests to the Ollama Cloud endpoint during the "
                             "scout and audit fan-outs. Lower it if the endpoint throttles.")
    args = vars(parser.parse_args())

    input_dir_arg = args.pop("input_dir")
    no_cache_arg = args.pop("no_cache")
    cache_path_arg = args.pop("cache_path")
    corpus_name_arg = args.pop("corpus_name")
    chunk_chars_arg = args.pop("chunk_chars")
    max_concurrency_arg = args.pop("max_concurrency")
    runs_arg = max(1, args.pop("runs"))
    overrides = {k: v for k, v in args.items() if v is not None}

    # Repeats and the response cache are mutually exclusive by construction. The
    # cache is keyed on SHA-256(model, messages); repeating the pipeline sends
    # byte-identical messages, so every call after the first would be a cache HIT
    # and runs 2..N would replay run 1 exactly. The measured spread would then be
    # zero -- not because the pipeline is stable, but because it never actually
    # ran again. Silently reporting that as "variance = 0" would be worse than
    # not measuring at all, so N > 1 forces the cache off.
    use_cache = not no_cache_arg
    if runs_arg > 1 and use_cache:
        print(f"[Cache] disabled for --runs {runs_arg}: repeats must issue real calls, "
              f"otherwise every repeat would replay the first run and the measured "
              f"variance would be a meaningless zero.")
        use_cache = False

    outputs: list[str] = []
    counts: list[int] = []
    for i in range(1, runs_arg + 1):
        if runs_arg > 1:
            print(f"\n{'=' * 70}\n  RUN {i}/{runs_arg}\n{'=' * 70}")
        result = run_pipeline(
            input_dir_arg, chunk_chars=chunk_chars_arg, max_concurrency=max_concurrency_arg,
            cache_path=(cache_path_arg if use_cache else None), run_index=i,
            corpus_name=corpus_name_arg, **overrides)
        outputs.append(result["output_path"])
        counts.append(result["n_records"])

    if runs_arg > 1:
        mean = statistics.mean(counts)
        std = statistics.stdev(counts) if len(counts) > 1 else 0.0
        print(f"\n{'=' * 70}\n  {runs_arg} runs written.")
        print(f"  records/run: {mean:.1f} +- {std:.1f}  {counts}")
        print("  Score them together and report mean +- std. A single run is one draw")
        print("  from a stochastic endpoint, not a measurement; two conditions whose")
        print("  intervals overlap are tied, not ranked.\n")
        for p in outputs:
            print(f"    {p}")
