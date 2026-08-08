# Superseded results — recall-only numeric term

**These are NOT the reported results.** Everything in this directory was produced
by the earlier form of `_numeric_overlap`, which scored only the ground truth's
recall over numbers and priced a record's surplus numbers at zero. It is kept
because it is the state the thesis reported before 2026-08-08, and because a
reader checking the metric change should be able to see both sides.

The current results live one directory up. The change, and every affected figure
old vs new, is documented in `../METRIC_CHANGE_2026-08-08.md`.

## Why the term changed

The recall-only form was exploitable. Appending to every predicted record the
numbers already present in *other records of the same file* — changing no
extracted content and using no knowledge of the ground truth — raised the score
on all four corpora, by +0.021 (biogas) to +0.178 (sulfuric acid). The symmetric
form (harmonic mean of that recall and the matching precision over the record's
own numbers) turns the same padding into a penalty of 0.030 to 0.177.

## Reproducing these files

The predictions are unchanged; only the scoring differs. Pass
`--numeric-recall-only` to `step3_evaluation_generic_dynamic.py` and every figure
here is reproduced exactly:

    python3 layers/layer_3/step3_evaluation_generic_dynamic.py \
        --gt data/external_test_biogas/ground_truth_biogas.csv \
        --pred-dir <staged biogas predictions> \
        --numeric-recall-only --aggregate

Headline figures under this metric: dev 0.719 ± 0.000, biogas 0.725 ± 0.012,
desalination 0.827 ± 0.035, sulfuric acid 0.642 ± 0.043.
