#!/bin/zsh
# Re-score the layer-2 grid-search runs with the SAME evaluator and settings used
# for the multi-agent pipeline, so the two are directly comparable.
#
# The grid CSVs are normalised into a staging directory first. They carry per-run
# bookkeeping columns (model_name, paradigm, level, run_id, llm_turns,
# text_truncated) that the generic evaluator has no reason to know about, and
# _pred_blob concatenates every unrecognised column into the content blob. Left
# in, every prediction would be scored with "gemma4:31b | cot_basic | 1 | ... |
# False" glued onto its text, depressing semantic similarity against the ground
# truth and making the grid look worse than it is. They are dropped here, and
# ruleId is renamed to id so exact-id matching works as it does for the pipeline.
# Nothing that carries extracted content is touched.
#
# Usage: zsh layers/evaluate_grid.sh

set -e
ROOT="${0:A:h:h}"
SRC="$ROOT/layers/layer_2/step2_results"
STAGE="$ROOT/.grid_stage"
GT="$ROOT/data/dataset/kg_seed/ground_truth.csv"

rm -rf "$STAGE"; mkdir -p "$STAGE"

python3 - "$SRC" "$STAGE" <<'PY'
import csv, os, sys

src, stage = sys.argv[1], sys.argv[2]
DROP = {"model_name", "paradigm", "level", "run_id", "llm_turns", "text_truncated"}
n = 0
for name in sorted(os.listdir(src)):
    if not name.endswith(".csv"):
        continue
    with open(os.path.join(src, name), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    cols = [c for c in rows[0] if c not in DROP]
    out_cols = ["id" if c == "ruleId" else c for c in cols]
    with open(os.path.join(stage, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({("id" if c == "ruleId" else c): r.get(c, "") for c in cols})
    n += 1
print(f"[stage] normalised {n} grid CSV(s) -> {stage}")
PY

python3 "$ROOT/layers/layer_3/step3_evaluation_generic_dynamic.py" \
    --gt "$GT" --pred-dir "$STAGE" --aggregate

rm -rf "$STAGE"
echo "################ GRID EVALUATION DONE -- see layers/step3_results ################"
