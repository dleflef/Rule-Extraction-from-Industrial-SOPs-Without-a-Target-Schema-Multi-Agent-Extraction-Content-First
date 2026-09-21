# How records are compared and how F1_content is computed

Every number in `RESULTS.csv` can be traced to a single comparison decision in
`match_audit/<corpus>/match_audit_<run>.csv`. This note says what those decisions
are; the CSVs show every one of them.

## 1. Records are NOT compared word by word

A predicted record and a ground-truth row are not required to share vocabulary,
and no field is aligned to any other field by name. The reason is structural: the
extractor discovers its own categories and field names per document
(`StaticLimit`, `crit_lo`, `warn_hi`, ...), while each ground truth uses its own fixed
vocabulary (`NormalOperatingLimit`, `bound_low`, `bound_high`, ...). Nothing ties
the two vocabularies together, so a word-level or field-level comparison would
score a correct extraction as wrong purely for naming things differently.

Instead each row on both sides is collapsed into **one text blob** — every column's
value joined with ` | ` — and the two blobs are compared as text. The audit CSV
prints the exact strings that were compared, in `gt_text_compared` and
`pred_text_compared`, so nothing about the comparison is hidden.

Two kinds of column are dropped before the blob is built, because they are
bookkeeping rather than rule content: the id column (reported only as the
id-reuse diagnostic) and provenance columns (source filename, source span — a filename match is not
evidence of a content match). Both are found structurally, not from a name list.

## 2. The comparison score for one pair

For a ground-truth row `g` and a predicted record `p`:

```
score(g, p) = 0.6 * semantic_cosine(g, p) + 0.4 * numeric_overlap(g, p)
```

- **semantic_cosine** — cosine similarity of SBERT (`all-MiniLM-L6-v2`) sentence
  embeddings of the two blobs. This is what lets `EnvironmentalLimit ... TMP ...
  175 195` line up with `ThresholdRule ... TMP ... 175.0 195.0` despite neither
  side knowing the other's vocabulary.
- **numeric_overlap** — numeric agreement is symmetric. In the GT-to-prediction
  direction, every GT number receives the score of its closest predicted number;
  in the prediction-to-GT direction, every predicted number receives the score
  of its closest GT number. Each closest-pair score is
  `max(0, 1 - |g - p| / max(|g|, |p|))`. The two directional means are combined
  by their harmonic mean. The first direction rewards recovery; the second
  charges unsupported surplus quantities. Sentence embeddings barely
  distinguish `175` from `195`, so without this term a record with the wrong
  bounds would score almost as well as the right one.
- If a GT row contains **no numbers at all** the numeric term is undefined rather
  than zero, and the score is the semantic cosine alone. The audit marks these
  rows: `numeric_overlap` is blank and `score_formula` reads `semantic only`.

Audit columns: `semantic_cosine`, `numeric_overlap`, `combined_score`, and
`score_formula`, which writes the arithmetic out literally
(`0.6*0.487 + 0.4*0.900 = 0.652`).
`numeric_recall_gt_found` and `numeric_precision_pred_justified` expose the two
directional means. `numeric_detail_gt_to_pred` and
`numeric_detail_pred_to_gt` show every closest-value pairing used to obtain
them (`50->40 (0.800)`).

For reference only, the audit also reports a **word-level** view of the same two
blobs — `word_overlap_jaccard_DIAGNOSTIC`, `shared_words_DIAGNOSTIC`,
`gt_only_words_DIAGNOSTIC`, `pred_only_words_DIAGNOSTIC`. **None of these enter
the score.** They are printed because "is this word by word?" is the first
question anyone asks of a semantic metric, and the answer is only checkable next
to the word-level picture it is being distinguished from.

## 3. Which record is compared against which row

Not all-against-all, and not first-against-first. The `n_gt x n_pred` matrix of
pair scores is solved as a **global optimal assignment** (Hungarian algorithm,
`scipy.optimize.linear_sum_assignment`), maximising the total score over all
one-to-one pairings. Each ground-truth row is therefore credited to at most one
predicted record and vice versa — one good record cannot be counted as satisfying
three different rules.

The assignment produces `min(n_gt, n_pred)` pairs. Whatever is left over is
unpaired by construction and appears in the audit as `UNPAIRED GT` (counts FN) or
`UNPAIRED RECORD` (counts FP).

Note the assignment maximises the *total* score, not the *count of pairs above
threshold*; these can differ in principle, and the choice is the conventional one.

**Reading `runner_up_score` and `margin_over_runner_up`.** The audit reports, for
every ground-truth row, the best score it could have got from any *other* record.
A negative margin is expected, not a defect: because the solver optimises the
total, a row sometimes yields its personally-best record to another row that
needs it more. Across all 20 runs, 81.8% of assigned rows got their own
best-scoring partner and 18.2% yielded it. A margin near zero (16.6% of pairs
are within ±0.05)
means the pairing was close to a coin toss and the pair should be read as such,
however comfortably it cleared the threshold.

**Precision has a ceiling set by the record count.** Only `min(n_gt, n_pred)`
pairs can exist, so when a run emits more records than the ground truth has rows,
the surplus are false positives whether or not they are correct. On
`dev_production_line`, every reported multi-agent run emits 131 records against
86 ground-truth rows, which caps precision at **0.656**. Each run has 78 accepted
pairs, 53 false positives and 8 false negatives, giving precision 0.595, recall
0.907 and F1 0.719. Those figures must be read against the record-count ceiling,
not against a hypothetical precision of 1.0. The
cap is reported as `max_precision` in `RESULTS.csv`, as `max_possible_precision`
per run, and as a warning on the console.

## 4. Threshold, then F1

A pair counts as a true positive when `combined_score >= 0.6`. Pairs below it are
shown in the audit as `REJECTED (below threshold)` and count as a false negative
for that GT row and a false positive for that record.

```
TP = assigned pairs scoring >= 0.6
FP = records_extracted - TP
FN = gt_rows_total    - TP
precision = TP/(TP+FP)     recall = TP/(TP+FN)     F1 = 2PR/(P+R)
```

`match_audit/<corpus>/f1_derivation_<corpus>.csv` carries these counts and the
substituted arithmetic per run, one row per run, alongside the parameters used
(threshold, weights, embedding model).

**F1_content is the only F1 reported.** Two earlier variants were removed rather
than kept as alternatives. A span-collapsed F1 merged records sharing a source
element, but the recorded span carries a per-item counter that made its grouping
key unique on every row, so it never merged anything and only ever duplicated
F1_content; repairing it would have made the granularity convention a per-corpus
choice worth ±0.2 F1, which is a selection knob, not a metric. An id-match F1 is
now reported as the **id-reuse diagnostic** (`id_reuse_frac`) instead: it counts
how many of the ground truth's identifiers appear verbatim among predicted ids,
which is a property of the annotator's convention — a ground truth using invented
row numbers (`R01`, `M001`) scores 0 no matter how good the extraction is, one
reusing printed codes (`OP-ST-001`) scores well. It is not an accuracy figure and
is not expressed as precision/recall/F1, because next to F1_content it would read
as one.

## 5. Free parameters

`0.6` (threshold), `0.6/0.4` (weights) and the embedding model are human choices.
They are conventional starting points, not values fitted to any ground truth here,
and all are overridable from the command line. `--sensitivity` re-scores across a
grid of thresholds and weightings and reports whether the ranking of systems is
stable, so a result that depends on the constants is visible as such.

Every constant is declared in `EvalConfig` — including the provenance-detection
thresholds (`noise_distinct_floor`, `noise_distinct_frac`, `noise_max_value_len`)
and `drop_tag_derived_numbers`. That last one is worth naming: the number regex
reads the hyphen in an asset tag as a minus sign, so `GSH-401` contributes `-401`
on both sides of the comparison, and such values are 15% of all numbers extracted
from these ground truths. They are counted by default, because they are genuine
asset-identity evidence and because excluding them *changes* reported figures:
development is unchanged, biogas moves +0.010, desalination +0.098, and sulfuric
acid -0.048. The flag exists so the choice is visible and reversible rather than
buried in a regex.

## 6. Does the metric actually measure extraction quality?

Tested, not asserted — `python3 layers/layer_3/step3_validity_checks.py`, which
writes `validity/` beside these files. Three results decide how the headline
figures should be read.

**Null case — predictions scored against a different corpus's ground truth.**
Nothing can be correct, so this is the floor the metric cannot go below on genre
similarity alone. The twelve directed wrong-corpus pairings have mean F1 values
from 0.000 to 0.075, well below the real 0.649–0.804 results.

**Perturbations — mean F1 lost when predictions are degraded:**

| perturbation | F1 lost | reading |
|---|---|---|
| standalone numeric quantities transformed by `x -> 3x + 7` | **0.080** | incorrect quantities reduce the score |
| records padded with other records' quantities | **0.101** | unsupported surplus quantities reduce the score |
| numeric positions reversed within applicable records | **0.000** | numeric slot binding is not captured |

The metric responds to wrong and unsupported quantities, but retaining the same
numeric multiset while reversing its positions leaves F1 unchanged. Read
`F1_content` as evidence of content presence, not as certification that every
quantity occupies the correct field. Field placement is checked separately.

**Threshold.** The reported operating point of 0.6 is a strict local maximum on
**none** of the four corpora — declining through it on development, biogas and
sulfuric acid, and on a plateau for desalination. A threshold chosen to flatter
the system would be a peak;
none is. The largest movement around 0.6 is **0.020 per 0.05 step**. The ranking
of corpora is not stable across the sweep, so no claim that one corpus is harder
than another is supported.

## 7. Reproducing the audit

```
python3 layers/layer_3/step3_evaluation_generic_dynamic.py \
    --gt <ground_truth.csv> --pred-dir <dir of run CSVs> \
    --aggregate --audit --label <corpus>
```

or `zsh layers/run_and_evaluate.sh 5 --eval-only` for all four corpora at once.
The audit is written from the *same* assignment object the metric is computed
from, not a second implementation, so it cannot drift from the reported figure.
