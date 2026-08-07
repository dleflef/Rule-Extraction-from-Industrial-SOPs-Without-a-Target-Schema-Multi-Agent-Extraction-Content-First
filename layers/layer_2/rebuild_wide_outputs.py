"""rebuild_wide_outputs.py
==================================================
Re-render existing extraction outputs in the flat, one-column-per-field shape
that to_rows now produces, without re-running any language model.

The pipeline writes two files per run: the scored CSV and a long-format
provenance sidecar (`facts_*.csv`) holding one row per (record, field, value).
The sidecar is written from the same assembled records as the CSV and BEFORE any
field is consumed for a fixed column, so it carries the complete field set. A
wide CSV is therefore fully recoverable from the sidecar plus the bookkeeping
columns of the existing CSV, and re-extraction would only spend LLM calls to
reproduce facts already on disk.

Field names are normalised by the same syntactic rule the pipeline applies, and
columns are ordered by the same frequency-descending, alphabetically-tie-broken
rule, so a rebuilt file is byte-identical in shape to one the pipeline would
write today. Nothing consults a ground truth: the column set is whatever the
corpus itself exhibited.

Usage:
    python3 rebuild_wide_outputs.py                 # rewrite in place
    python3 rebuild_wide_outputs.py --dry-run       # report, change nothing
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2_multi_agent_generic import normalise_field_name  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "step2_results_generic")
FIXED_HEAD, FIXED_TAIL = ["id", "category"], ["source_file", "source_span"]


def facts_path_for(pred_path: str) -> str:
    """The sidecar written by the same run as this prediction file."""
    base = os.path.basename(pred_path)
    return os.path.join(os.path.dirname(pred_path),
                        re.sub(r"^ext_multi_agent_generic_", "facts_", base))


def rebuild(pred_path: str) -> pd.DataFrame:
    old = pd.read_csv(pred_path).fillna("")
    facts = pd.read_csv(facts_path_for(pred_path)).fillna("")

    # Bookkeeping travels with the record; content comes from the sidecar.
    keep = [c for c in FIXED_HEAD + FIXED_TAIL if c in old.columns]
    book = old[keep].copy()

    values: dict[str, dict[str, str]] = {}
    for _, f in facts.iterrows():
        rid, name, value = str(f["record_id"]), f["field"], f["value"]
        if value in (None, "") or str(value).strip() == "":
            continue
        col = normalise_field_name(name)
        cell = values.setdefault(rid, {})
        prev = str(cell.get(col, "")).strip()
        cell[col] = value if not prev else (
            f"{prev} | {value}" if str(value) not in prev else prev)

    counts: dict[str, int] = {}
    for cell in values.values():
        for col in cell:
            counts[col] = counts.get(col, 0) + 1
    discovered = sorted(counts, key=lambda k: (-counts[k], k))
    columns = [c for c in FIXED_HEAD if c in book.columns] + discovered + \
              [c for c in FIXED_TAIL if c in book.columns]

    rows = []
    for _, b in book.iterrows():
        cell = values.get(str(b["id"]), {})
        rows.append({c: (b[c] if c in book.columns else cell.get(c, "")) for c in columns})
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.results_dir,
                                          "ext_multi_agent_generic_*.csv")))
    if not preds:
        print(f"No prediction files in {args.results_dir}")
        raise SystemExit(1)

    print(f"{'file':<62}{'old cols':>9}{'new cols':>9}{'rows':>7}")
    changed = 0
    for p in preds:
        if not os.path.exists(facts_path_for(p)):
            print(f"  [skip] {os.path.basename(p)}: no facts sidecar")
            continue
        old_cols = len(pd.read_csv(p).columns)
        wide = rebuild(p)
        print(f"{os.path.basename(p)[:60]:<62}{old_cols:>9}{len(wide.columns):>9}{len(wide):>7}")
        if not args.dry_run:
            wide.to_csv(p, index=False)
            changed += 1
    print(f"\n{'(dry run, nothing written)' if args.dry_run else f'rewrote {changed} file(s)'}")
