#!/bin/zsh
# Held-out evaluation on the desalination corpus: both systems, then scoring.
#
#   Phase 1  multi-agent pipeline, 5 runs
#   Phase 2  grid search, gemma4:31b x all paradigms x 5 runs
#   Phase 3  score both against ground_truth_desalination.csv
#
# --gt-id-field identifier is passed explicitly. This ground truth leaves
# `identifier` blank on the rows that have no printed code (matrix cells, un-coded
# prose rules), and the evaluator's structural id detection reads a missing value
# as the string "nan", counts it as present, and therefore rejects `identifier`
# for low uniqueness -- leaving `rule_text` as the only candidate. Detecting
# rule_text as the id would drop the verbatim rule sentences out of the ground
# truth's content blob, which is the richest signal it has.
#
# Usage: zsh layers/run_desalination.sh [runs]     (default 5)

set -e
ROOT="${0:A:h:h}"
RUNS=${1:-5}
CORPUS="$ROOT/data/external_test_desalination"
GT="$CORPUS/ground_truth_desalination.csv"
STAGE="$ROOT/.desal_stage"

echo "################ PHASE 1: multi-agent ($RUNS runs) ################"
cd "$ROOT/layers/layer_2"
python3 step2_multi_agent_generic.py --input-dir "$CORPUS" --runs "$RUNS"

echo "################ PHASE 2: grid search (gemma4:31b, all paradigms) ################"
python3 step2_grid_search_extraction_en.py --texts-dir "$CORPUS" \
    --models gemma4:31b --runs "$RUNS"

echo "################ PHASE 3: evaluation ################"
rm -rf "$STAGE"; mkdir -p "$STAGE/multi_agent" "$STAGE/grid"

cp "$ROOT/layers/layer_2/step2_results_generic"/ext_multi_agent_generic_external_test_desalination_run*.csv \
   "$STAGE/multi_agent/" 2>/dev/null || echo "  (no multi-agent results)"

# Grid CSVs carry per-run bookkeeping columns the generic evaluator would fold
# into the content blob, and name their id column ruleId. Normalise both, exactly
# as layers/evaluate_grid.sh does, so grid and pipeline are scored alike.
python3 - "$ROOT/layers/layer_2/step2_results" "$STAGE/grid" <<'PY'
import csv, os, sys
src, stage = sys.argv[1], sys.argv[2]
DROP = {"model_name", "paradigm", "level", "run_id", "llm_turns", "text_truncated"}
n = 0
for name in sorted(os.listdir(src)):
    if not name.endswith(".csv") or "desalination" not in name:
        continue
    with open(os.path.join(src, name), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    cols = [c for c in rows[0] if c not in DROP]
    out = ["id" if c == "ruleId" else c for c in cols]
    with open(os.path.join(stage, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({("id" if c == "ruleId" else c): r.get(c, "") for c in cols})
    n += 1
print(f"[stage] normalised {n} grid CSV(s)")
PY

for sys_name in multi_agent grid; do
  if [ -n "$(ls -A $STAGE/$sys_name 2>/dev/null)" ]; then
    echo "############ $sys_name ############"
    python3 "$ROOT/layers/layer_3/step3_evaluation_generic_dynamic.py" \
        --gt "$GT" --pred-dir "$STAGE/$sys_name" --gt-id-field identifier --aggregate
  fi
done
rm -rf "$STAGE"
echo "################ DESALINATION DONE ################"
