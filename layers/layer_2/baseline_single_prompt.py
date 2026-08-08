"""baseline_single_prompt.py
==================================================
Schema-free single-prompt baseline: what one LLM call per document achieves
when nothing about the target schema is supplied.

WHY THIS EXISTS. The thesis reports the multi-agent pipeline against two
configured conditions -- a schema-specified prompting grid and a hand-written
regex parser -- and both are told things about the corpus that the pipeline is
not. Neither therefore isolates what the ORCHESTRATION buys. A committee's
obvious question is whether the LangGraph machinery (segmentation with line
numbering, parallel scouts, the corpus-level schema inducer, deterministic
assembly, the grounding audit) improves anything over simply handing the whole
document to the same model with the same extraction philosophy. This condition
answers that, and it is the only condition in this project that differs from the
pipeline in ARCHITECTURE ALONE.

WHAT IS HELD CONSTANT. Everything that is not orchestration:
  - the same model (gemma4:31b), endpoint, temperature, seed, retry policy and
    output budget, through the pipeline module's own llm_call;
  - the same extraction philosophy, expressed in the scout prompt's own words:
    the document decides what one record is, field names are copied from the
    document's own labels, values are copied verbatim, no worked example is
    given, and a functional `category` is asked for;
  - the same JSON reply contract and the same tolerant parser;
  - the same evaluator, ground truths, repeat count and scoring parameters.

WHAT IS REMOVED -- and this is the whole of the manipulation:
  - segmentation and line numbering (Stage 1): the document is passed whole, so
    there are no chunks and no line numbers to cite. The `lines` field and the
    "[Section: ...]" marker convention go with them, since both are constructs
    of the chunker rather than of the document.
  - the parallel scout fan-out (Stage 2): one call per document, not one per
    chunk.
  - the schema inducer (Stage 3): no corpus-level vocabulary reconciliation, so
    whatever names the model produces for a document are the names that reach
    the output.
  - deterministic assembly (Stage 4): no merging of records sharing a printed
    identifier and no canonical renaming. One returned JSON object becomes one
    row.
  - the grounding audit (Stage 5): nothing re-reads a record against its source.

Serialising JSON objects to a rectangular CSV still requires taking the union of
the keys observed, and that is done here. It is not schema induction: no name is
changed, merged, or reconciled, and a record simply leaves blank the columns it
did not use. Column ordering follows the pipeline's own convention
(frequency-descending, alphabetical tie-break) so the two outputs are laid out
comparably.

HOW TO READ THE RESULT, decided before it was run. If the baseline scores lower,
the orchestration is load-bearing. If it scores the same or higher, that is
reported as it stands: agentic orchestration costs calls without buying accuracy
when no schema is supplied. Both outcomes are publishable and neither is the
hoped-for one.

Usage:
    python3 baseline_single_prompt.py --input-dir <dir> --corpus-name <tag> --runs 5
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step2_multi_agent_generic as P   # noqa: E402  (endpoint, llm_call, parsers)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "baseline_single_prompt_results")


_CITE_CLAUSE = """

Every content line of the document below is numbered (L7, L8, ...). Also give
each record:
  "lines" — the numbers of the lines this record was read from, as integers.
  Cite only line numbers printed in the document below. A record read from one
  table row cites that row's line; a rule stated across two lines cites both."""


def baseline_prompt(cite_lines: bool = False) -> str:
    """The scout prompt with the orchestration-dependent passages removed.

    Kept verbatim wherever the wording does not depend on chunking: the
    record-boundary rules, the attribute-vs-case column test, the field-naming
    discipline, the value-copying discipline, the category request, and the
    explicit refusal to give a worked example. Removing more would make this a
    comparison between two prompts rather than between two architectures;
    keeping the chunk-specific passages would ask the model to cite line
    numbers that do not exist.
    """
    return """You are a precision data-extraction engineer reading ONE complete
industrial operating or control document. You are not told in advance how the
plant, its processes, or its vocabulary are organised — every document names
the things, people, places, processes and codes it governs in its own words,
and you must not assume any of that vocabulary going in. Extract every
explicitly stated rule, limit, requirement, or procedure.

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
  - If ONE passage states SEVERAL rules — because it prints several
    identifiers, or because it states genuinely different consequences or
    importance levels for different situations — emit one record per rule. Do
    not bundle them, and do not split a single rule into pieces.

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
and every row of it inherits it. Whenever a record's own text does not name the
entity it concerns but the heading governing it does, copy that entity into a
field of the record, named after whatever the document calls that kind of
entity. A record that cannot say what it is about is not usable, so this is not
optional: check every record you emit for a field naming its subject before you
return it.

But NEVER create a field for the document's own title, code, revision, or the
name of the facility the whole document is about. Such a value belongs to every
record equally, so it distinguishes nothing and is recorded separately as
provenance. The test is simple: if a field would hold the same value for every
record in the document, it is not a property of any rule and must be left out.

Also give each record:
  "category" — what the rule FUNCTIONALLY IS, in two or three words of your
  own (a standing cap, an automatic reaction, a rule about who may enter, a
  recurring duty...), judged by what it does and never by where on the page it sits: a
  table and a paragraph can state the same kind of rule.

Scan the WHOLE document systematically — every section, every line, every row,
every cell — and do not stop after the first record. Extract every distinct
rule, including ones that state only a bare limit, status, or classification
with no consequence attached; deciding what is significant enough to keep is not
this pass's job. Skip only literal non-content: a repeated column header, a page
footer, a document title.

No worked example is given, deliberately. Any example would demonstrate one
particular document shape — a tiered numeric table, a cross-reference matrix,
a recurring schedule — and a model shown such an example reproduces its
vocabulary and its structure on documents that have neither. The only thing to work from is the
document itself.

Reply EXCLUSIVELY with JSON:
{"records": [{"id": "<printed identifier or empty>", "category": "<what it is>",
%s"fields": {"<field_name>": "<value>", ...}}, ...]}
You may prefix it with a brief "reasoning" key.""" % ('"lines": [<numbers>], ' if cite_lines else "") \
        + (_CITE_CLAUSE if cite_lines else "") + P._COT_INSTRUCTION


def _numbered(text: str) -> str:
    """The document with every non-blank line numbered, using the pipeline's own
    convention (L<n>, n = 1-indexed raw file line) so a citation from this
    condition and a citation from the pipeline mean exactly the same thing and
    can be checked by the same code."""
    return "\n".join(f"L{i}: {ln.rstrip()}"
                     for i, ln in enumerate(text.split("\n"), start=1) if ln.strip())


def extract_document(fname: str, text: str, model: str,
                      cite_lines: bool = False) -> list[dict]:
    """One LLM call for one whole document."""
    body = _numbered(text) if cite_lines else text
    messages = [
        {"role": "system", "content": baseline_prompt(cite_lines)},
        {"role": "user", "content": f"DOCUMENT: {fname}\n\n{body}"},
    ]
    raw = P.llm_call(model, messages)
    items = P.parse_items(raw, "records")
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        fields = it.get("fields")
        if not isinstance(fields, dict):
            fields = {}
        cited = it.get("lines") if cite_lines else None
        cited = [int(n) for n in cited if str(n).strip().lstrip("-").isdigit()] \
            if isinstance(cited, list) else []
        out.append({
            "id": str(it.get("id", "") or "").strip(),
            "category": str(it.get("category", "") or "").strip(),
            "fields": {str(k): v for k, v in fields.items() if v not in (None, "")},
            "source_file": fname,
            "lines": cited,
        })
    return out


def to_rows(records: list[dict]) -> list[dict]:
    """Rectangular CSV, union of observed keys. No renaming, no merging.

    Column ordering copies the pipeline's convention (frequency-descending,
    alphabetical tie-break) so the two outputs are laid out the same way and a
    reader comparing them is not also comparing two layouts.
    """
    used: set[str] = set()
    seq: dict[str, int] = {}
    rows = []
    for r in records:
        fname = r["source_file"]
        own = r["id"]
        if own and own not in used:
            rid = own
        else:
            seq[fname] = seq.get(fname, 0) + 1
            stem = "".join(ch for ch in os.path.splitext(fname)[0] if ch.isalnum())[:12].upper()
            rid = f"{stem}-{seq[fname]:03d}"
        used.add(rid)
        row = {"id": rid, "category": r["category"]}
        for name, value in r["fields"].items():
            col = P.normalise_field_name(name)
            if value in (None, ""):
                continue
            prev = str(row.get(col, "")).strip()
            row[col] = value if not prev else (
                f"{prev} | {value}" if str(value) not in prev else prev)
        row["source_file"] = fname
        if r.get("lines"):
            row["source_span"] = " | ".join(
                f"{fname}::L{n}#0" for n in r["lines"])[:200]
        rows.append(row)

    head, tail = ["id", "category"], ["source_file", "source_span"]
    counts: dict[str, int] = {}
    for r in rows:
        for k in r:
            if k not in head and k not in tail:
                counts[k] = counts.get(k, 0) + 1
    cols = head + sorted(counts, key=lambda k: (-counts[k], k)) + tail
    return [{c: r.get(c, "") for c in cols} for r in rows]


def run_once(input_dir: str, corpus: str, model: str, run_index: int,
             cite_lines: bool = False) -> str:
    import csv
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".txt"))
    if not files:
        raise SystemExit(f"no .txt documents in {input_dir}")
    records = []
    for fname in files:
        with open(os.path.join(input_dir, fname), encoding="utf-8") as fh:
            text = fh.read()
        print(f"    [{fname}] {len(text)} chars -> 1 call", flush=True)
        got = extract_document(fname, text, model, cite_lines)
        print(f"      {len(got)} record(s)", flush=True)
        records.extend(got)

    rows = to_rows(records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR,
                        f"ext_single_prompt{'_cited' if cite_lines else ''}_"
                        f"{corpus}_run{run_index}_{ts}.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(f"    -> {len(rows)} rows  {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--corpus-name", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--cite-lines", action="store_true",
                    help="number the content lines with the pipeline's own L<n> "
                         "convention and require each record to cite the lines it was "
                         "read from. Tests whether provenance needs the graph or only "
                         "the numbering.")
    args = ap.parse_args()

    # Repeats must issue real calls, for the same reason the pipeline disables
    # its cache under --runs: a warm cache would replay one run N times and
    # report a standard deviation of zero that reflects the cache rather than
    # the endpoint.
    P._CACHE = None
    print(f"[Cache] disabled -- {args.runs} repeat(s) must issue real calls")
    print(f"[Baseline] single prompt, one call per document, model={args.model}")

    for i in range(1, args.runs + 1):
        print(f"\n  ===== {args.corpus_name} run {i}/{args.runs} =====", flush=True)
        run_once(args.input_dir, args.corpus_name, args.model, i, args.cite_lines)
