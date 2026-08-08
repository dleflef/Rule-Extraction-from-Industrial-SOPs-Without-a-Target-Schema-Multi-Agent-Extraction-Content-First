#!/bin/zsh
# Score the schema-free single-prompt baseline with the SAME evaluator, ground
# truths and parameters used for the multi-agent pipeline, so the two rows are
# directly comparable in the results table.
#
# The baseline writes every corpus into one directory, and the evaluator's batch
# mode scores a whole directory against a single ground truth, so each corpus is
# staged on its own first -- the same reason run_and_evaluate.sh stages.
#
#   -> layers/step3_results/RESULTS_single_prompt_baseline.csv
#
# Usage:  zsh layers/evaluate_single_prompt_baseline.sh

set -e
ROOT="${0:A:h:h}"
EVAL="$ROOT/layers/layer_3/step3_evaluation_generic_dynamic.py"
SRC="$ROOT/layers/layer_2/baseline_single_prompt_results"
STAGE="$ROOT/.baseline_stage"
REPORT="$ROOT/layers/step3_results/RESULTS_single_prompt_baseline.csv"

# Same corpus table and same desalination id-field override as run_and_evaluate.sh.
CORPORA=(
  "dev_production_line:$ROOT/data/dataset/kg_seed/ground_truth.csv:-"
  "external_test_biogas:$ROOT/data/external_test_biogas/ground_truth_biogas.csv:-"
  "external_test_sulfuric_acid:$ROOT/data/external_test_sulfuric_acid/ground_truth_SA.csv:-"
  "external_test_desalination:$ROOT/data/external_test_desalination/ground_truth_desalination.csv:identifier"
)

rm -rf "$STAGE"; rm -f "$REPORT"

for entry in $CORPORA; do
  tag="${entry%%:*}"; rest="${entry#*:}"
  gt="${rest%%:*}"; idfield="${rest#*:}"
  mkdir -p "$STAGE/$tag"
  cp "$SRC"/ext_single_prompt_${tag}_run*.csv "$STAGE/$tag/" 2>/dev/null || {
    echo "  (no baseline results for $tag, skipping)"; continue; }
  idflag=()
  [[ "$idfield" != "-" ]] && idflag=(--gt-id-field "$idfield")
  echo "############ $tag ############"
  python3 "$EVAL" --gt "$gt" --pred-dir "$STAGE/$tag" $idflag \
      --aggregate --label "${tag}_singleprompt" --report "$REPORT"
done

rm -rf "$STAGE"
echo "################ DONE ################"
echo "Baseline table : $REPORT"
