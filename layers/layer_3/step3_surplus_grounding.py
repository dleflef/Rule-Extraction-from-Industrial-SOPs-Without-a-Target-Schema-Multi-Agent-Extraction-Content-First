"""step3_surplus_grounding.py
==================================================
Are the surplus records fabrications, or facts the annotation has no row for?

The granularity argument in the results rests on a claim about the records that
the assignment leaves unpaired: that they state things the source documents
contain, and are counted as false positives only because the annotator itemised
more coarsely. Stated that way it is an assertion, and "a substantial share"
is not a measurement. This script replaces it with a count.

The check is deterministic and uses no language model. Every record carries a
`source_span` naming the file and the 1-indexed raw line it was read from
(Stage 1 numbers the lines in code, which is what makes this checkable at all).
For each UNPAIRED record the script reads those lines back out of the source
document and asks whether the numeric values the record asserts actually appear
there. A record that asserts a threshold absent from its own cited line is a
hallucination; a record whose values are all present on the line it cites is a
fact the annotation did not record.

Two deliberate strictnesses, both of which make the reported figure harder to
reach rather than easier:

  - Only the lines the record ITSELF cites are consulted. A value appearing
    elsewhere in the document does not count. This is the anti-hallucination
    question ("did the model read this off the line it claims") and not the
    weaker question of whether the number exists somewhere in the corpus.
  - Tag-derived numbers are excluded by default. A hyphen inside an asset tag
    reads as a minus sign (see the evaluator's _extract_numbers), and those
    values match trivially on both sides because the tag is copied verbatim.
    Counting them would inflate the grounded fraction with identifiers rather
    than thresholds. --with-tag-numbers reports the looser figure too.

Numbers are extracted with the evaluator's own _extract_numbers, so "a number"
means the same thing here as it does in the metric, and compared as floats so
that 175 and 175.0 agree.

Records carrying no numbers at all cannot be checked this way and are reported
as a separate class rather than folded into either outcome, with a token-level
overlap against the cited lines given for orientation.

Usage:
    python3 step3_surplus_grounding.py                # all corpora
    python3 step3_surplus_grounding.py --corpus dev_production_line
"""

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_evaluation_generic_dynamic as E   # noqa: E402

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRED_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
AUDIT_DIR = os.path.join(E.STEP3_RESULTS_DIR, "match_audit")
OUT_DIR = os.path.join(E.STEP3_RESULTS_DIR, "robustness")

# corpus tag : directory holding the source documents the records cite
CORPORA = [
    ("dev_production_line",        os.path.join(_PROJECT_ROOT, "layers", "layer_1", "texts")),
    ("external_test_biogas",       os.path.join(_PROJECT_ROOT, "data", "external_test_biogas")),
    ("external_test_desalination", os.path.join(_PROJECT_ROOT, "data", "external_test_desalination")),
    ("external_test_sulfuric_acid", os.path.join(_PROJECT_ROOT, "data", "external_test_sulfuric_acid")),
]

# FILE::L<line>#<index>. The index is the record's ordinal within the line and is
# not needed here; the file and line are.
_SPAN_RE = re.compile(r"([\w.\-]+\.txt)::L(\d+)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Columns that are bookkeeping rather than asserted content. `lines` is included
# because it is the known auditor defect documented in the pipeline section: it
# holds a record's own cited line numbers, so counting it would test whether a
# line number appears on its own line, which is vacuously true.
_NOT_CONTENT = {"id", "source_file", "source_span", "chunk_id", "lines"}
# `category` is additionally excluded from the VALUE-level check below. The
# category label is the inducer's own vocabulary, derived from the corpus rather
# than copied from any line (that is the whole point of Stage 3), so it is not
# expected to appear in the source and counting it would penalise the pipeline
# for working as designed. It is a discovered label, not an asserted fact.
_NOT_A_QUOTED_FACT = _NOT_CONTENT | {"category"}


def _norm(s: str) -> str:
    """Loose comparison form: case-folded, whitespace-collapsed, unicode folded.
    Deliberately does NOT strip punctuation, so a value must still match the
    document's own characters."""
    return re.sub(r"\s+", " ", E._fold_unicode(str(s)).strip().lower())


def _source_lines(doc_dir: str) -> Dict[str, List[str]]:
    """Raw lines of every source document, keyed by basename. Raw, not
    content-filtered: the pipeline's L<n> is a 1-indexed raw file line."""
    out = {}
    for path in glob.glob(os.path.join(doc_dir, "*.txt")):
        with open(path, encoding="utf-8") as f:
            out[os.path.basename(path)] = f.read().split("\n")
    return out


def _cited_text(span: str, docs: Dict[str, List[str]]) -> Tuple[str, int]:
    """The verbatim source text of every line this record cites, concatenated.

    Read from the source FILE, never from the text echoed inside source_span
    itself -- the echoed copy is what the model reported, and checking a model's
    output against its own report would establish nothing.
    """
    parts, n = [], 0
    for fname, lineno in _SPAN_RE.findall(str(span or "")):
        lines = docs.get(fname)
        if not lines:
            continue
        i = int(lineno) - 1              # L<n> is 1-indexed
        if 0 <= i < len(lines):
            parts.append(lines[i])
            n += 1
    return " ".join(parts), n


def _content_numbers(row: Dict, drop_tags: bool) -> List[float]:
    vals = []
    for k, v in row.items():
        if k in _NOT_CONTENT:
            continue
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s:
            vals.append(s)
    return E._extract_numbers(E._fold_unicode(" | ".join(vals)), drop_tags)


def null_control(tag: str, doc_dir: str, foreign_dir: str) -> Tuple[int, int, int]:
    """Null case for this check, on the same principle as the metric's own.

    A substring test against a whole document could be vacuous: if any short
    value matches any document, "100% grounded" would mean nothing. The control
    scores the same values against a document they did NOT come from. A test
    that discriminates must collapse; one that does not is measuring the
    existence of common words.
    """
    docs = _source_lines(doc_dir)
    foreign = _norm(" ".join(l for ls in _source_lines(foreign_dir).values() for l in ls))
    audits = sorted(glob.glob(os.path.join(AUDIT_DIR, tag, "match_audit_*.csv")))
    own = fgn = tot = 0
    for apath in audits:
        audit = pd.read_csv(apath)
        ppath = os.path.join(PRED_DIR, os.path.basename(apath).replace("match_audit_", ""))
        if not os.path.exists(ppath):
            continue
        ids = set(audit[audit["decision"].astype(str)
                  .str.startswith("UNPAIRED RECORD")]["pred_id"].astype(str))
        for rec in pd.read_csv(ppath).fillna("").to_dict("records"):
            if str(rec.get("id", "")) not in ids:
                continue
            own_blob = _norm(" ".join(docs.get(str(rec.get("source_file", "")), [])))
            for k, v in rec.items():
                if k in _NOT_A_QUOTED_FACT or not str(v).strip():
                    continue
                n = _norm(v)
                tot += 1
                own += n in own_blob
                fgn += n in foreign
    return own, fgn, tot


def check_corpus(tag: str, doc_dir: str, drop_tags: bool = True) -> pd.DataFrame:
    docs = _source_lines(doc_dir)
    audits = sorted(glob.glob(os.path.join(AUDIT_DIR, tag, "match_audit_*.csv")))
    rows = []
    for apath in audits:
        audit = pd.read_csv(apath)
        run = re.search(r"(run\d+)", os.path.basename(apath))
        run = run.group(1) if run else os.path.basename(apath)
        # the prediction file this audit was produced from
        pname = os.path.basename(apath).replace("match_audit_", "")
        ppath = os.path.join(PRED_DIR, pname)
        if not os.path.exists(ppath):
            print(f"  [skip] no prediction file for {pname}")
            continue
        pred = pd.read_csv(ppath).fillna("")
        by_id = {str(r.get("id", "")): r for r in pred.to_dict("records")}

        unpaired = audit[audit["decision"].astype(str).str.startswith("UNPAIRED RECORD")]
        for _, a in unpaired.iterrows():
            rec = by_id.get(str(a.get("pred_id", "")))
            if rec is None:
                continue
            cited, n_lines = _cited_text(rec.get("source_span", ""), docs)
            nums = _content_numbers(rec, drop_tags)
            cited_nums = E._extract_numbers(E._fold_unicode(cited), drop_tags)
            found = [v for v in nums if any(abs(v - c) < 1e-9 for c in cited_nums)]
            # VALUE-level grounding: every field value the record asserts should
            # be quoted from the line it cites. This reaches every record, not
            # only the minority carrying numbers, and it is the check that
            # actually answers "is this a fact from the document or an
            # invention". A value counts as grounded when it appears as a
            # substring of the cited text under loose normalisation.
            cited_n = _norm(cited)
            # Document-level scope as well as line-level. The line-level test is
            # the strict one, but it is unfair to two legitimate cases: a matrix
            # record takes its subject from a COLUMN HEADER, which sits on a
            # different line from the cell it cites, and a chunk legitimately
            # spans several lines the scout could read. A value absent from the
            # whole document, by contrast, cannot have been read anywhere and is
            # the actual invention signal. Both are reported.
            doc_n = _norm(" ".join(docs.get(str(rec.get("source_file", "")), [])))
            values = [str(v).strip() for k, v in rec.items()
                      if k not in _NOT_A_QUOTED_FACT and str(v).strip()]
            grounded_vals = [v for v in values if _norm(v) and _norm(v) in cited_n]
            in_doc_vals = [v for v in values if _norm(v) and _norm(v) in doc_n]
            rec_toks = _TOKEN_RE.findall(
                E._fold_unicode(" ".join(str(v) for k, v in rec.items()
                                         if k not in _NOT_A_QUOTED_FACT and str(v).strip())).lower())
            cited_toks = set(_TOKEN_RE.findall(cited.lower()))
            tok_cov = (sum(1 for t in rec_toks if t in cited_toks) / len(rec_toks)) if rec_toks else 0.0
            rows.append({
                "corpus": tag, "run": run, "pred_id": a.get("pred_id", ""),
                "cited_lines": n_lines,
                "values_asserted": len(values),
                "values_quoted_from_cited_lines": len(grounded_vals),
                "values_present_in_source_document": len(in_doc_vals),
                "all_values_grounded": (len(values) > 0 and len(grounded_vals) == len(values)),
                "all_values_in_document": (len(values) > 0 and len(in_doc_vals) == len(values)),
                "numbers_asserted": len(nums),
                "numbers_found_on_cited_lines": len(found),
                "fully_grounded": (len(nums) > 0 and len(found) == len(nums)),
                "has_no_numbers": len(nums) == 0,
                "token_coverage_of_cited_lines": round(tok_cov, 3),
                "record_numbers": " ".join(f"{v:g}" for v in nums),
                "cited_line_numbers": " ".join(f"{v:g}" for v in cited_nums),
                "ungrounded_values": " || ".join(v for v in values if v not in grounded_vals)[:300],
                "values_absent_from_document": " || ".join(v for v in values if v not in in_doc_vals)[:300],
            })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"  {label}: no unpaired records")
        return
    n_rec = len(df)
    v_tot = int(df["values_asserted"].sum())
    v_ok = int(df["values_quoted_from_cited_lines"].sum())
    rec_all = int(df["all_values_grounded"].sum())
    numeric = df[~df["has_no_numbers"]]
    n_num = len(numeric)
    print(f"\n  {label}")
    print(f"    unpaired records                       : {n_rec}")
    print(f"    field values asserted by them          : {v_tot}")
    print(f"    ...quoted from their own cited line(s) : {v_ok}"
          + (f"  ({v_ok/v_tot:.1%})" if v_tot else ""))
    print(f"    records with EVERY value grounded      : {rec_all}"
          + (f"  ({rec_all/n_rec:.1%})" if n_rec else ""))
    d_ok = int(df["values_present_in_source_document"].sum())
    d_rec = int(df["all_values_in_document"].sum())
    print(f"    -- widened to the whole source document --")
    print(f"    values present somewhere in the document: {d_ok}"
          + (f"  ({d_ok/v_tot:.1%})" if v_tot else ""))
    print(f"    records with EVERY value in the document: {d_rec}"
          + (f"  ({d_rec/n_rec:.1%})" if n_rec else ""))
    if n_num:
        g = int(numeric["fully_grounded"].sum())
        nt = int(numeric["numbers_asserted"].sum())
        nk = int(numeric["numbers_found_on_cited_lines"].sum())
        print(f"    of these, records asserting a number   : {n_num}"
              f"  -- all numbers present: {g}/{n_num}, values {nk}/{nt}")
    else:
        print(f"    of these, records asserting a number   : 0 "
              f"(surplus here is textual, not numeric)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="", help="restrict to one corpus tag")
    ap.add_argument("--with-tag-numbers", action="store_true",
                    help="also count values produced by reading a hyphen in an asset "
                         "tag as a minus sign (looser; off by default)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    print("=" * 78)
    print("  Are the surplus (unpaired) records grounded in the lines they cite?")
    print("=" * 78)

    targets = [(t, d) for t, d in CORPORA if not args.corpus or t == args.corpus]
    all_rows = []
    for tag, doc_dir in targets:
        df = check_corpus(tag, doc_dir, drop_tags=not args.with_tag_numbers)
        summarise(df, tag)
        all_rows.append(df)
        if args.with_tag_numbers:
            continue
        loose = check_corpus(tag, doc_dir, drop_tags=False)
        if not loose.empty:
            ln = loose[~loose["has_no_numbers"]]
            if len(ln):
                print(f"    [incl. tag-derived values            : "
                      f"{int(ln['fully_grounded'].sum())}/{len(ln)}]")

    # Null control: the same values scored against a document they did not come
    # from. Reported next to the figure it qualifies, never separately.
    print("\n  " + "-" * 74)
    print("  NULL CONTROL (values scored against a document they did not come from)")
    for tag, doc_dir in targets:
        foreign = next(d for t, d in CORPORA if t != tag)
        own, fgn, tot = null_control(tag, doc_dir, foreign)
        if tot:
            print(f"    {tag:<30} own {own}/{tot} ({own/tot:.1%})"
                  f"   foreign {fgn}/{tot} ({fgn/tot:.1%})")
    print("    A grounded fraction should be read against the foreign figure, which is")
    print("    what short or generic values match by coincidence, not against zero.")

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not out.empty:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "surplus_grounding.csv")
        out.to_csv(path, index=False)
        print(f"\n  Per-record detail written to: {path}")
        print("  Deterministic: no model is called, so re-running reproduces this exactly.")
