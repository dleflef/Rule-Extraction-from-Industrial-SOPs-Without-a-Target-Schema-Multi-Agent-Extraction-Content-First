"""step3_schema_stability.py
==================================================
Stability of the induced corpus schema across repeated runs.

The inducer synthesises one corpus-level schema per run from that run's scout
observations, and nothing in F1_content rewards it: the metric flattens field
names away before scoring. Whether the inducer behaves consistently is
therefore a separate empirical question, and repetition already answers part
of it: the production protocol ran every corpus five times, and each run saved
its induced schema. If the arbiter were hallucinating or merging fields
erratically, five independent inductions over the same documents would not
agree on the canonical vocabulary; if they agree, the induction is at least a
stable function of the corpus rather than of one sample of endpoint noise.

Stability is reported per corpus as: the canonical field-set size per run,
the mean pairwise Jaccard overlap between the runs' field sets, the fraction
of the union vocabulary present in every run, the same three figures for
categories, and how many distinct choices the runs made for each induced role
pointer (1 = every run picked the same field). A companion detail file lists
every canonical field with the number of runs that induced it, so unstable
names can be inspected individually.

Nothing here re-runs or re-judges anything: every figure is a count over the
induced_schema_*.json files the pipeline already saved.

Writes step3_results/schema_stability.csv
and    step3_results/schema_stability_fields.csv.

Usage:
    python3 step3_schema_stability.py
"""

import glob
import itertools
import json
import os
import re
import sys

import pandas as pd

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCHEMA_DIR = os.path.join(_PROJECT_ROOT, "layers", "layer_2", "step2_results_generic")
OUT_PATH = os.path.join(_PROJECT_ROOT, "layers", "step3_results",
                        "schema_stability.csv")
DETAIL_PATH = os.path.join(_PROJECT_ROOT, "layers", "step3_results",
                           "schema_stability_fields.csv")

_STEM = re.compile(r"induced_schema_(?P<corpus>.+)_run(?P<run>\d+)_\d{8}_\d{6}\.json$")
_ROLES = ("condition_field", "action_field", "severity_field")


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _set_stats(sets: list[set]) -> tuple[float, float, int, int]:
    """(mean pairwise jaccard, fraction of union present in all runs,
    union size, core size)."""
    pairs = list(itertools.combinations(sets, 2))
    mean_j = (sum(_jaccard(a, b) for a, b in pairs) / len(pairs)) if pairs else 1.0
    union = set().union(*sets)
    core = set.intersection(*sets) if sets else set()
    return mean_j, (len(core) / len(union) if union else 1.0), len(union), len(core)


def main() -> None:
    paths = sorted(glob.glob(os.path.join(SCHEMA_DIR, "**", "induced_schema_*.json"),
                             recursive=True))
    by_corpus: dict[str, list[dict]] = {}
    for path in paths:
        m = _STEM.search(os.path.basename(path))
        if not m:
            continue
        with open(path, encoding="utf-8") as f:
            by_corpus.setdefault(m.group("corpus"), []).append(json.load(f))
    if not by_corpus:
        sys.exit(f"no induced_schema files under {SCHEMA_DIR}")

    summaries, details = [], []
    for corpus, schemas in sorted(by_corpus.items()):
        field_sets = [set(s.get("fields", {})) for s in schemas]
        cat_sets = [set(s.get("categories", {})) for s in schemas]
        f_j, f_core_frac, f_union, f_core = _set_stats(field_sets)
        c_j, c_core_frac, c_union, c_core = _set_stats(cat_sets)
        sizes = [len(fs) for fs in field_sets]
        summaries.append({
            "corpus": corpus,
            "runs": len(schemas),
            "fields_per_run_min": min(sizes),
            "fields_per_run_max": max(sizes),
            "field_union": f_union,
            "field_core": f_core,
            "field_core_frac": round(f_core_frac, 3),
            "field_jaccard_mean": round(f_j, 3),
            "category_union": c_union,
            "category_core": c_core,
            "category_core_frac": round(c_core_frac, 3),
            "category_jaccard_mean": round(c_j, 3),
            **{f"{role}_distinct": len({s.get(role, "") for s in schemas})
               for role in _ROLES},
        })
        for name in sorted(set().union(*field_sets)):
            details.append({"corpus": corpus, "field": name,
                            "runs_present": sum(name in fs for fs in field_sets),
                            "runs_total": len(schemas)})

    pd.DataFrame(summaries).to_csv(OUT_PATH, index=False)
    pd.DataFrame(details).to_csv(DETAIL_PATH, index=False)

    for r in summaries:
        print(f"  {r['corpus']}: {r['runs']} runs, {r['fields_per_run_min']}-"
              f"{r['fields_per_run_max']} fields/run, core {r['field_core']}/"
              f"{r['field_union']} ({r['field_core_frac']:.0%} of union), "
              f"jaccard {r['field_jaccard_mean']:.3f}, roles distinct "
              + "/".join(str(r[f"{role}_distinct"]) for role in _ROLES))
    print(f"[schema-stability] wrote {OUT_PATH}")
    print(f"[schema-stability] wrote {DETAIL_PATH}")


if __name__ == "__main__":
    main()
