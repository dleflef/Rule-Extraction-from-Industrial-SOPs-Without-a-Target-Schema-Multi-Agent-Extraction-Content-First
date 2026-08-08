#!/bin/zsh
# The two scoring-parameter sweeps behind the robustness artifacts, re-scoring
# the frozen predictions locally (no LLM endpoint needed).
#
#   Sweep 1  --drop-tag-derived-numbers: rescore every corpus with the declared
#            toggle flipped, so the reported figures can be shown to be the
#            conservative setting rather than the flattering one.
#            -> layers/step3_results/robustness/RESULTS_drop_tag_numbers.csv
#   Sweep 2  --sensitivity: threshold x semantic-weight grid per corpus, so the
#            operating point can be quoted with its curve.
#            -> layers/step3_results/robustness/sensitivity_<corpus>.csv
#   Sweep 3  --no-unicode-fold: rescore with blob normalisation disabled, so the
#            figures reported under the default can be read against the figures
#            that the same predictions produce without it.
#            -> layers/step3_results/robustness/RESULTS_no_unicode_fold.csv
#
# Usage:  zsh layers/run_robustness_sweeps.sh

set -e
ROOT="${0:A:h:h}"
EVAL="$ROOT/layers/layer_3/step3_evaluation_generic_dynamic.py"
RESULTS="$ROOT/layers/layer_2/step2_results_generic"
STEP3="$ROOT/layers/step3_results"
DEST="$STEP3/robustness"
STAGE="$ROOT/.robustness_stage"
REPORT="$DEST/RESULTS_drop_tag_numbers.csv"
NOFOLD="$DEST/RESULTS_no_unicode_fold.csv"

# Same corpus table as run_and_evaluate.sh: tag : ground truth : id-field override.
CORPORA=(
  "dev_production_line:$ROOT/data/dataset/kg_seed/ground_truth.csv:-"
  "external_test_biogas:$ROOT/data/external_test_biogas/ground_truth_biogas.csv:-"
  "external_test_sulfuric_acid:$ROOT/data/external_test_sulfuric_acid/ground_truth_SA.csv:-"
  "external_test_desalination:$ROOT/data/external_test_desalination/ground_truth_desalination.csv:identifier"
)

mkdir -p "$DEST"
rm -rf "$STAGE"
rm -f "$REPORT" "$NOFOLD"

for entry in $CORPORA; do
  tag="${entry%%:*}"; rest="${entry#*:}"
  gt="${rest%%:*}"; idfield="${rest#*:}"
  mkdir -p "$STAGE/$tag"
  cp "$RESULTS"/ext_multi_agent_generic_${tag}_run*.csv "$STAGE/$tag/"
  idflag=()
  [[ "$idfield" != "-" ]] && idflag=(--gt-id-field "$idfield")

  echo "############ $tag : tag-number toggle ############"
  python3 "$EVAL" --gt "$gt" --pred-dir "$STAGE/$tag" $idflag \
      --drop-tag-derived-numbers --aggregate \
      --label "${tag}_notag" --report "$REPORT"
  # The toggled per-run/summary files land in step3_results with a _notag label;
  # move them beside the report so the default-configuration artifacts there are
  # never mixed with toggled ones.
  mv "$STEP3"/{runs,summary}_${tag}_notag_*.csv "$DEST/" 2>/dev/null || true

  echo "############ $tag : normalisation off ############"
  python3 "$EVAL" --gt "$gt" --pred-dir "$STAGE/$tag" $idflag \
      --no-unicode-fold --aggregate \
      --label "${tag}_nofold" --report "$NOFOLD"
  rm -f "$STEP3"/{runs,summary}_${tag}_nofold_*.csv

  echo "############ $tag : threshold x weight sweep ############"
  python3 "$EVAL" --gt "$gt" --pred-dir "$STAGE/$tag" $idflag \
      --sensitivity --sens-thresholds 0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9
  mv "$(ls -t "$STEP3"/sensitivity_*.csv | head -1)" "$DEST/sensitivity_${tag}.csv"
done

rm -rf "$STAGE"
echo "################ DONE ################"
echo "Toggle table : $REPORT"
echo "Fold-off     : $NOFOLD"
echo "Sweeps       : $DEST/sensitivity_<corpus>.csv"
