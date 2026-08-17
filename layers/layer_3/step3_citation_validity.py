"""step3_citation_validity.py
==================================================
Are the pipeline's citations checkable, and are they correct?

Every record the multi-agent pipeline emits cites the numbered line it was read
from. A citation that cannot be resolved, or that points at a line not containing
what the record asserts, is worse than no citation at all, because it looks
auditable and is not. This script tests both properties against the source
documents.

A citation is only worth something if it is
CHECKABLE and CORRECT, so three things are counted per condition:

  1. COVERAGE  -- does the record cite anything at all?
  2. VALIDITY  -- does every cited line number exist in the document named?
                  A citation to a line that is not there is worse than none,
                  because it looks auditable and is not.
  3. FIDELITY  -- do the record's own field values actually appear on the lines
                  it cites? This is the question an auditor asks: the operator
                  clicks the citation and must see the value there.

Fidelity is the deciding measure and is scored the same way as the surplus
grounding check (step3_surplus_grounding.py): a value counts when it appears as
a substring of the cited text under loose normalisation, with the discovered
category label excluded because it is the inducer's vocabulary rather than a
quotation. The same null control applies -- values are also scored against a
document they did not come from, so a fidelity figure can be read against what
coincidence alone achieves.

The pipeline is expected to do well on 1 and 2 by construction: its scouts can
only cite line numbers the harness printed in the chunk in front of them. That
is precisely the claim being tested -- whether a constraint enforced by the
harness is worth anything over an instruction followed by the model.

Usage:
    python3 step3_citation_validity.py
"""

import glob
import os
import re
import sys
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_evaluation_generic_dynamic as E          # noqa: E402
import step3_surplus_grounding as G                   # noqa: E402

_PROJECT_ROOT = G._PROJECT_ROOT
PIPE_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
OUT = os.path.join(E.STEP3_RESULTS_DIR, "robustness", "citation_validity.csv")

CONDITIONS = [
    ("multi_agent",    PIPE_DIR, "ext_multi_agent_generic_{tag}_run*.csv"),
]


def _pipeline_lines(pred_path: str) -> Dict[str, List[tuple]]:
    """Resolve a pipeline record's cited lines from its facts sidecar.

    Necessary for a fair comparison, and the reason is a formatting detail that
    would otherwise be read as a finding. A pipeline record's `source_span`
    begins with its BOUNDARY KEY, and when the document printed an identifier
    for the rule that key is the identifier (`SOP_001...txt::RULE-ST01-01`)
    rather than a line reference. Parsing line numbers out of the span alone
    therefore reports "no citation" for every record whose rule the document
    numbered -- 36 of 131 on one development run -- when those records are in
    fact the BEST provenanced of the set. The sidecar written beside every run
    carries the actual line numbers per record, so it is used instead.
    """
    facts = pred_path.replace("ext_multi_agent_generic_", "facts_")
    if not os.path.exists(facts):
        return {}
    out: Dict[str, List[tuple]] = {}
    for r in pd.read_csv(facts).fillna("").to_dict("records"):
        rid = str(r.get("record_id", ""))
        fname = str(r.get("source_file", ""))
        for n in str(r.get("lines", "")).split(","):
            n = n.strip()
            if n.isdigit():
                out.setdefault(rid, [])
                if (fname, n) not in out[rid]:
                    out[rid].append((fname, n))
    return out


def analyse(tag: str, doc_dir: str, files: List[str]) -> Dict:
    docs = G._source_lines(doc_dir)
    foreign_dir = next(d for t, d in G.CORPORA if t != tag)
    foreign = G._norm(" ".join(l for ls in G._source_lines(foreign_dir).values() for l in ls))

    n_rec = n_cited = n_valid = 0
    v_tot = v_ok = v_fgn = 0
    bad_lines = 0
    for path in files:
        sidecar = _pipeline_lines(path)
        for rec in pd.read_csv(path).fillna("").to_dict("records"):
            n_rec += 1
            span = str(rec.get("source_span", ""))
            refs = sidecar.get(str(rec.get("id", "")), []) or G._SPAN_RE.findall(span)
            if not refs:
                continue
            n_cited += 1
            ok_all = True
            for fname, lineno in refs:
                lines = docs.get(fname)
                if not lines or not (1 <= int(lineno) <= len(lines)):
                    ok_all = False
                    bad_lines += 1
            if ok_all:
                n_valid += 1
            cited = " ".join(
                docs[f][int(n) - 1] for f, n in refs
                if f in docs and 1 <= int(n) <= len(docs[f]))
            cited_n = G._norm(cited)
            for k, v in rec.items():
                if k in G._NOT_A_QUOTED_FACT or not str(v).strip():
                    continue
                nv = G._norm(v)
                if not nv:
                    continue
                v_tot += 1
                v_ok += nv in cited_n
                v_fgn += nv in foreign
    return {
        "corpus": tag, "records": n_rec,
        "cited_any": n_cited, "coverage": round(n_cited / n_rec, 3) if n_rec else 0.0,
        "all_lines_exist": n_valid,
        "validity": round(n_valid / n_cited, 3) if n_cited else 0.0,
        "dangling_line_refs": bad_lines,
        "values_checked": v_tot,
        "values_on_cited_lines": v_ok,
        "fidelity": round(v_ok / v_tot, 3) if v_tot else 0.0,
        "fidelity_null_control": round(v_fgn / v_tot, 3) if v_tot else 0.0,
    }


if __name__ == "__main__":
    print("=" * 90)
    print("  Citation coverage, validity and fidelity: does provenance need the graph?")
    print("=" * 90)
    rows = []
    for cond, d, pat in CONDITIONS:
        print(f"\n  {cond}")
        print(f"    {'corpus':<30}{'recs':>6}{'cover':>8}{'valid':>8}{'dangling':>10}"
              f"{'fidelity':>10}{'(null)':>9}")
        for tag, doc_dir in G.CORPORA:
            files = sorted(glob.glob(os.path.join(d, pat.format(tag=tag))))
            if not files:
                continue
            r = analyse(tag, doc_dir, files)
            r["condition"] = cond
            rows.append(r)
            print(f"    {tag:<30}{r['records']:>6}{r['coverage']:>8.3f}{r['validity']:>8.3f}"
                  f"{r['dangling_line_refs']:>10}{r['fidelity']:>10.3f}"
                  f"{r['fidelity_null_control']:>9.3f}")
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        df.to_csv(OUT, index=False)
        print(f"\n  Written: {OUT}")
        print("  coverage = records citing anything; validity = of those, all cited lines exist;")
        print("  fidelity = record values found on the lines cited; null = same values against")
        print("  a foreign document, i.e. what coincidence alone achieves.")
