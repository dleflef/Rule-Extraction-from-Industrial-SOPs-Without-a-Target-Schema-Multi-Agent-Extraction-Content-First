#!/bin/zsh
# Instrumented audit-stage replication.
#
# The production run's artifacts are all post-audit: nothing in them records
# what the auditor kept, dropped, or corrected, so the audit stage's measured
# contribution cannot be recovered from them. This script repeats the exact
# production protocol (same corpora, same models, same flags, 5 repeats per
# corpus) with the audit-verdict sidecar in place, so every auditor decision is
# written to audit_log_*.csv and the stage's intervention rate becomes a
# reportable number. It is a REPLICATION, not the production run: its scores
# are reported only to show the replication behaves comparably, and every
# output carries an `_auditrep` label so nothing it writes can be confused
# with, grouped with, or overwrite the frozen production artifacts.
#
# Usage:  zsh layers/run_audit_replication.sh [runs]     (default 5)

set -e
ROOT="${0:A:h:h}"
RUNS=${1:-5}

RESULTS="$ROOT/layers/layer_2/step2_results_generic"
STAGE="$ROOT/.eval_stage_auditrep"
DEST_L2="$RESULTS/audit_replication"
DEST_L3="$ROOT/layers/step3_results/robustness/audit_replication"
REPORT="$DEST_L3/AUDIT_REPLICATION_RESULTS.csv"

# Same corpus table as run_and_evaluate.sh, tags suffixed with _auditrep.
CORPORA=(
  "dev_production_line_auditrep:$ROOT/layers/layer_1/texts:$ROOT/data/dataset/kg_seed/ground_truth.csv:-"
  "external_test_biogas_auditrep:$ROOT/data/external_test_biogas:$ROOT/data/external_test_biogas/ground_truth_biogas.csv:-"
  "external_test_sulfuric_acid_auditrep:$ROOT/data/external_test_sulfuric_acid:$ROOT/data/external_test_sulfuric_acid/ground_truth_SA.csv:-"
  "external_test_desalination_auditrep:$ROOT/data/external_test_desalination:$ROOT/data/external_test_desalination/ground_truth_desalination.csv:identifier"
)

mkdir -p "$DEST_L2" "$DEST_L3"

echo "################ PHASE 1: replication extraction ($RUNS runs per corpus) ################"
cd "$ROOT/layers/layer_2"
for entry in $CORPORA; do
  tag="${entry%%:*}"; rest="${entry#*:}"; indir="${rest%%:*}"
  echo "############ $tag ############"
  python3 step2_multi_agent_generic.py --input-dir "$indir" \
      --corpus-name "$tag" --runs "$RUNS"
done

echo "################ PHASE 2: replication scoring ################"
rm -rf "$STAGE"
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
        --gt "$gt" --pred-dir "$STAGE/$tag" --aggregate \
        --label "$tag" --report "$REPORT"
  else
    python3 "$ROOT/layers/layer_3/step3_evaluation_generic_dynamic.py" \
        --gt "$gt" --pred-dir "$STAGE/$tag" --gt-id-field "$idfield" --aggregate \
        --label "$tag" --report "$REPORT"
  fi
done
rm -rf "$STAGE"

# Quarantine every replication artifact away from the frozen production files.
mv "$RESULTS"/*_auditrep_*.csv "$RESULTS"/*_auditrep_*.json "$DEST_L2/" 2>/dev/null || true
mv "$ROOT/layers/step3_results"/*auditrep*.csv "$DEST_L3/" 2>/dev/null || true

echo "################ DONE ################"
echo "Audit verdicts : $DEST_L2/audit_log_*_auditrep_*.csv"
echo "Scores         : $REPORT"
