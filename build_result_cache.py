"""
Build step2_result_cache.json from existing genuine CSV results.

Run this ONCE after placing your original step2 result CSVs in:
    layers/layer_2/step2_results/          (baseline, ablations, grid search)
    layers/layer_2/step2_results/multi_run/ (20 per-run CSVs)

After this script runs, the pipeline will restore those exact files from the
cache on every re-run — even if the CSVs are deleted — without touching the LLM.

Usage (from the project root):
    $env:PYTHONUTF8 = "1"
    python build_result_cache.py
"""
import os
import sys

_ROOT    = os.path.dirname(os.path.abspath(__file__))
_LAYER2  = os.path.join(_ROOT, "layers", "layer_2")
_RESULTS = os.path.join(_LAYER2, "step2_results")
_MULTI   = os.path.join(_RESULTS, "multi_run")
_CACHE   = os.path.join(_RESULTS, "step2_result_cache.json")

sys.path.insert(0, _LAYER2)
from llm_cache import ResultCache

cache = ResultCache(_CACHE)

added = 0
skipped = 0


def ingest(directory: str) -> None:
    global added, skipped
    if not os.path.isdir(directory):
        print(f"  [skip] directory not found: {directory}")
        return
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        if cache.get(fname) == content:
            skipped += 1
            continue
        cache.set(fname, content)
        added += 1
        print(f"  cached  {fname}  ({len(content)} chars)")


print(f"Reading CSVs from {_RESULTS} ...")
ingest(_RESULTS)

print(f"Reading CSVs from {_MULTI} ...")
ingest(_MULTI)

print(f"\nDone. {added} added, {skipped} already identical.")
print(f"Cache: {len(cache)} total entries  ->  {_CACHE}")
print("\nYou can now delete any step2 CSV and re-run the pipeline to get it back identically.")
