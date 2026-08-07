#!/bin/zsh
# End-to-end layer-2 -> layer-3 measurement.
#
#   Phase 1  run step2_multi_agent_generic.py over every corpus, N repeats each.
#   Phase 2  score each corpus against its OWN ground truth in the evaluator's
#            batch mode, which writes evaluation_summary_generic_dynamic_<ts>.csv
#            and aggregate_by_condition_<ts>.csv into layers/step3_results/.
#
# Phase 2 stages one directory per corpus before scoring. The evaluator's batch
# mode scores every CSV in a directory against a single ground truth, so pointing
# it straight at step2_results_generic would score each corpus against the wrong
# GT and would also try to score the facts_*.csv provenance sidecars as if they
# were predictions.
#
# Usage:  zsh layers/run_and_evaluate.sh [runs]        (default 5)
#         zsh layers/run_and_evaluate.sh 5 --eval-only  (skip phase 1)

set -e
ROOT="${0:A:h:h}"
RUNS=${1:-5}
EVAL_ONLY=${2:-}

RESULTS="$ROOT/layers/layer_2/step2_results_generic"
STAGE="$ROOT/.eval_stage"

# corpus tag : input dir : ground truth : id-field override ("-" = auto-detect)
#
# desalination overrides the id field. Its ground truth leaves `identifier` blank
# on rows with no printed code (matrix cells, un-coded prose rules), and the
# evaluator's structural detection reads a missing value as the string "nan",
# counts it as present, and rejects `identifier` for low uniqueness -- leaving
# `rule_text` as the only candidate. Detecting rule_text as the id would drop the
# verbatim rule sentences out of the ground truth's content blob. The other five
# ground truths populate their id column on every row and detect correctly.
# The first entry is the DEVELOPMENT corpus: the four Production Line A SOPs the
# pipeline was built and debugged against. Its directory is still called "texts"
# because layer 1, run_pipeline.py and the grid all write to or read from that
# path by name; the label is set with --corpus-name instead, so results tables
# say what the corpus is and what role it played rather than "texts".
CORPORA=(
  "dev_production_line:$ROOT/layers/layer_1/texts:$ROOT/data/dataset/kg_seed/ground_truth.csv:-"
  "external_test_biogas:$ROOT/data/external_test_biogas:$ROOT/data/external_test_biogas/ground_truth_biogas.csv:-"
  "external_test_cleanroom:$ROOT/data/external_test_cleanroom:$ROOT/data/external_test_cleanroom/ground_truth_cleanroom.csv:-"
  "external_test_desalination:$ROOT/data/external_test_desalination:$ROOT/data/external_test_desalination/ground_truth_desalination.csv:identifier"
)

if [[ "$EVAL_ONLY" != "--eval-only" ]]; then
  echo "################ PHASE 1: extraction ($RUNS runs per corpus) ################"
  cd "$ROOT/layers/layer_2"
  for entry in $CORPORA; do
    tag="${entry%%:*}"; rest="${entry#*:}"; indir="${rest%%:*}"
    echo "############ $tag ############"
    python3 step2_multi_agent_generic.py --input-dir "$indir" \
        --corpus-name "$tag" --runs "$RUNS"
  done
fi

echo "################ PHASE 2: evaluation ################"
rm -rf "$STAGE"
# One combined table across all corpora. Cleared first so a re-run replaces it
# rather than appending a second copy of every corpus.
REPORT="$ROOT/layers/step3_results/RESULTS.csv"
rm -f "$REPORT"
for entry in $CORPORA; do
  tag="${entry%%:*}"; rest="${entry#*:}"; rest="${rest#*:}"
  gt="${rest%%:*}"; idfield="${rest#*:}"
  mkdir -p "$STAGE/$tag"
  cp "$RESULTS"/ext_multi_agent_generic_${tag}_run*.csv "$STAGE/$tag/" 2>/dev/null || {
    echo "  (no results for $tag, skipping)"; continue; }
  echo "############ $tag ############"
  if [[ "$idfield" == "-" ]]; then
    python3 "$ROOT/layers/layer_3/step3_evaluation_generic_dynamic.py" \
        --gt "$gt" --pred-dir "$STAGE/$tag" --aggregate --audit \
        --label "$tag" --report "$REPORT"
  else
    python3 "$ROOT/layers/layer_3/step3_evaluation_generic_dynamic.py" \
        --gt "$gt" --pred-dir "$STAGE/$tag" --gt-id-field "$idfield" --aggregate --audit \
        --label "$tag" --report "$REPORT"
  fi
done
rm -rf "$STAGE"
echo "################ DONE ################"
echo "Combined table : layers/step3_results/RESULTS.csv"
echo "Per-corpus     : layers/step3_results/summary_<corpus>_<ts>.csv (mean/std)"
echo "Per-run detail : layers/step3_results/runs_<corpus>_<ts>.csv"
