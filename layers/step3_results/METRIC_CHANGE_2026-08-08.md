# Numeric term made symmetric — every affected figure, old vs new

**Change.** `_numeric_overlap` scored only GT-recall: "does the record contain each of
the ground truth's numbers?" It never asked whether the record's own numbers were
justified. Surplus values were therefore free. It is now the harmonic mean of that
recall and the matching precision over the record's numbers (`EvalConfig.numeric_symmetric`,
default on; `--numeric-recall-only` reproduces the old behaviour).

**Why it mattered.** Appending to every predicted record the numbers already present in
*other records of the same file* — changing no extracted content and using no knowledge of
the ground truth — raised the old figure on all four corpora. The fix turns that gain into
a penalty:

| corpus | padding gain, old term | padding gain, new term |
|---|---|---|
| dev production line | **+0.065** | −0.177 |
| biogas | **+0.021** | −0.059 |
| desalination | **+0.151** | −0.030 |
| sulfuric acid | **+0.178** | −0.121 |

Nothing was re-extracted. The evaluator is deterministic and all 20 retained prediction
files were rescored; `--numeric-recall-only` reproduces every previously reported figure
exactly.

---

## 1. Main results (Tables `main_results`, `app_multiagent`)

| corpus | F1 old | F1 new | P old | P new | R old | R new |
|---|---|---|---|---|---|---|
| dev production line | 0.719 ± 0.000 | **0.719 ± 0.000** | 0.595 | 0.595 | 0.907 | 0.907 |
| biogas | 0.725 ± 0.012 | **0.715 ± 0.023** | 0.971 | 0.957 | 0.578 | 0.570 |
| RO desalination | 0.827 ± 0.035 | **0.850 ± 0.019** | 0.809 | 0.831 | 0.846 | 0.869 |
| sulfuric acid | 0.642 ± 0.043 | **0.649 ± 0.049** | 0.779 | 0.788 | 0.546 | 0.552 |

Record counts, GT counts and precision ceilings are unchanged. The dev corpus is still
identical across all five repeats. The corpus range becomes 0.649–0.850 (was 0.642–0.827);
the dev corpus is still neither best nor worst.

## 2. Schema-specified grid (Tables `grid_contrast`, `app_grid`)

Largest move of any condition: −0.019. Ranking of the `gemma4:31b` conditions is unchanged.

| condition | old | new |
|---|---|---|
| gemma4-31b reflexion_guided | 0.946 | **0.949** |
| gemma4-31b reflexion | 0.939 | 0.939 |
| gemma4-31b pre_act | 0.896 | **0.894** |
| gemma4-31b cot_basic | 0.895 | **0.892** |
| baseline_b0 (regex parser) | 0.771 | 0.771 |

`cot_basic` and `pre_act` remain tied, so the scout-paradigm selection rationale stands.
In the `nemotron` tail three conditions move by −0.015 to −0.019 and reorder among
themselves.

Derived claims: schema specification is now worth **0.230** F1 on the dev corpus
(0.949 − 0.719), was 0.227. Grid-vs-regex gap becomes **0.178** (was 0.175).

## 3. Metric validity (Table `perturbations`, null case, threshold sweep)

**Null case — substantially stronger.** Predictions scored against a foreign corpus's
ground truth:

| | old | new |
|---|---|---|
| per-corpus pooled | 0.000 – 0.193 | **0.014 – 0.037** |
| worst single pairing | 0.193 | **0.101** |

**Perturbations (mean F1 lost):**

| perturbation | old | new | reading |
|---|---|---|---|
| every numeric value corrupted | 0.218 | **0.201** (0.085–0.324) | values are verified |
| bounds reversed within record | 0.000 | **0.000** | slot binding still not checked |
| numerics permuted between records | −0.008 | **−0.001** | still not checked |
| category label permuted | −0.000 | **+0.007** | still ~unpriced |
| token order scrambled (vs control) | +0.002 | **0.000** | order still not read |

The blind spots are unchanged — this fix does not address slot binding.

**Threshold sweep.** The operating point τ = 0.6 is still a local maximum on **no** corpus.
All four curves are now *declining* through it (biogas and sulfuric acid were previously
flat there), so the "insensitive on two corpora" claim no longer holds and should be
dropped. Largest movement per 0.05 step: 0.023 (desalination); dev 0.015, biogas 0.016,
sulfuric 0.020. At τ = 0.4: 0.793 / 0.747 / 0.917 / 0.810.

## 4. Floors and rescaling (`lim_floor`)

Per-GT floor = highest score any foreign corpus's predictions reach against it.

| ground truth | floor old | floor new | reported | rescaled onto [floor, 1] |
|---|---|---|---|---|
| dev production line | 0.193 | **0.059** | 0.719 | 0.701 |
| biogas | 0.092 | **0.101** | 0.715 | 0.683 |
| sulfuric acid | 0.171 | **0.050** | 0.649 | 0.631 |
| desalination | 0.052 | **0.027** | 0.850 | 0.846 |

Adjustments remain downward throughout, but are now −0.004 to −0.032 (were −0.007 to
−0.072). "Read against a floor of roughly 0.19" becomes roughly **0.10**.

## 5. Other affected figures

- **Granularity swing (`granularity`)**: regrouped sulfuric acid annotation gives
  **0.796 ± 0.065** (was 0.788 ± 0.058), recall 0.796. The swing is **0.147** (was 0.146) —
  the headline granularity finding is unchanged.
- **Tag-derived-number toggle (`numeric`, `lim_threshold`)**: dev unmoved (0.719);
  biogas now +0.010 (0.715 → 0.725, previously unmoved); sulfuric −0.048 (0.649 → 0.601,
  was −0.037); desalination +0.067 (0.850 → 0.917, was +0.090).
- **Unicode fold**: still inert. Disabling it reproduces 0.719 / 0.715 / 0.850 / 0.649
  exactly.
- **Audit trail stats (`audit_trail`)**: 945 assigned pairs; **82.5%** of rows receive their
  own best-scoring record, **17.5%** yield to the global optimum, **16.8%** sit within
  ±0.05 of the runner-up. Assignments became more decisive.
- **Binding check (`binding_check`)**: **unchanged**. biogas `ll` 26/26 and `hl` 10/10;
  desalination 20/20 each; dev 600/920 present with 420 (0.70) in the correct slot;
  desalination 50/50 with 40 (0.80).
- **Field verification (`field_verification`)**: metric-independent, unaffected.

## 6. Audit-trail columns

Two columns added so the trail remains a printout of the number actually used:
`numeric_recall_gt_found` and `numeric_precision_pred_justified`, plus
`numeric_detail_pred_to_gt` for the reverse per-number pairing. `f1_derivation_*.csv`
now records `numeric_term` (`symmetric` / `gt_recall_only`).

---

## Pre-existing staleness found while verifying (NOT caused by this change)

Both of these were already inconsistent with the retained artifacts *before* the metric
was touched, and both are reviewer-checkable:

1. **The worked example** describes biogas run 1 as producing **37 records** and derives
   TP = 33, FP = 4, FN = 14, P = 0.892, R = 0.702, F1 = 0.786. The retained run 1 has
   **28 records**. It also quotes a cosine of 0.855 for the `R04` pair (retained value:
   0.8359) and presents `R15` as `REJECTED` at 0.286 against record `MID-036`; in the
   retained batch `R15` was already a true positive at 0.8817 against `MID-008`. Under the
   new term that pair scores 0.8386 and `R04` scores 0.870
   (0.6 × 0.836 + 0.4 × 0.921, with numeric recall 1.000 and precision 0.854).
   The section states it is "reproducible by opening that file", so it needs rebuilding
   from a current audit row.

2. **The audit-trail statistics** are quoted as 927 assigned pairs with 70% / 30% /
   18%. The pre-change artifact already read 945 pairs and 77.1% / 22.9% / 16.3%.
