# Phase 2 Evaluation — Limitations and Validity Notes

Honest accounting of what the Phase 2 detection numbers show, written to be
adapted into the thesis' threats-to-validity section. All numbers come from
`detection_results/` (regenerable end-to-end with `run_all.sh`; engine in
step5_core.py; evaluated against the iMAKS guide v3, Zenodo record
20075430).

## Headline results

| run | recall | 95% CI | precision | notes |
|---|---|---|---|---|
| Original 14 GT events (dev-exposed) | 0.929 (13/14) | [0.685, 0.987] | 0.867 | GT-0009 is the one GAP |
| Excl. GT-0009 (guide §3.3 scores it separately) | 1.000 (13/13) | [0.772, 1.000] | — | |
| Held-out injection, single one-shot seed (42 events) | 0.976 (41/42) | [0.877, 0.996] | 0.854 | landed at the TOP of the seed distribution — do not quote alone |
| **Held-out injection, POOLED over 10 pre-committed seeds (420 events)** | **0.907 (381/420)** | **[0.876, 0.931]** | **0.880** | **the honest headline number** |

The multi-seed pooled estimate is the number to lead with. Per-seed recall
ranges 0.810–0.976 (`holdout_multiseed.csv`), which shows a single
42-event draw carries ±0.08 of schedule luck; seed 20260710's 0.976 was
the luckiest region of that range. Apples-to-apples ordering is as
expected for honest evaluation: dev-exposed 13/13 (1.000) > held-out
pooled 0.907. Pooled per type: STUCK 80/80, SPIKE 98/100,
OUT_OF_RANGE 122/140, DRIFT 81/100 — the slow-ramp DRIFT weakness seen in
the strictness curve is confirmed out-of-sample. Pooled recall under the
stricter ≥25%-overlap criterion: 0.795.

**GT-0009 / CORRELATED scope decision:** no multi-source fusion detector is
implemented. The guide (§2.3.2) defines the CORRELATED type as requiring
three-way evidence fusion (timeseries co-occurrence + SOP-001 causal rule +
SOP-003 corroboration); this stage of the work deliberately covers only the
four single-sensor detectors (threshold, stuck, drift, sustained), so
GT-0009 is scored by the same uniform rule as every other event, shows as
GAP, and the guide's separate GT-0009 binary is reported as NOT ATTEMPTED.
Fusion is future work; the extraction gap it would depend on (SOP-003 §2
"Correlated Fault Resolution" table, currently not extracted) is noted
under Phase 1 improvements.

## What is verified clean (no data leakage) — an executable proof

- **`step5_leakage_audit.py` (→ `leakage_audit.csv`, verdict PASS):**
  timeseries_raw.csv physically contains GT-derived columns (nominal,
  warn_hi, crit_hi, warn_lo, crit_lo) in the rows the detector streams.
  The audit removes all 13 non-{timestamp,sensor_id,value} columns and
  reruns the full detection path: violation rates, the quarantine set, and
  all 1,153 alarms are bit-identical. It also verifies no Rule node in
  Neo4j carries GT anomaly fields (gtId/anomalyType/startTs/endTs), and
  that detection completes before load_gt_windows() is ever called. GT
  enters at scoring time only — as a runtime result, not a code-reading
  claim.
- The `RULE-THR-*` family matches GT bounds exactly because it was
  extracted from `SOP_002_AlarmThresholds.txt`, which contains those
  thresholds by dataset design — correct extraction, not leakage (20 of 44
  extracted threshold rules elsewhere carry visibly wrong bounds).
- `step5_baselines.py` rebuilds the rule set from step4 CSVs without Neo4j
  and reproduces the Neo4j-driven result exactly (tp=13, fp=2).

## Limitation 1 — Design-loop exposure, and its (partial) fix

The detector stack was developed while observing outcomes on all 14
original GT events, INCLUDING the nominal test split — so 13/14 (and test
5/5) is dev-exposed, not a generalization estimate. One specific disclosed
instance: the per-sensor drift-window sizing was fixed after observing that
a fixed 60-min window made MAINT-06 ("drift >2°C over 2 h") structurally
unable to fire.

**Fix implemented (two layers, both in `step5_holdout.py`):**

1. Default one-shot mode — 42 new anomaly instances per schedule,
   following the dataset's own generative definitions (guide §2.3.2:
   SPIKE Gaussian transient peaking in the first 20% of the window, DRIFT
   monotonic ramp, STUCK frozen at onset, OUT_OF_RANGE fixed offset
   beyond WARN) and magnitude scales (§2.3.1), placed at seeded-random
   unseen times, run through the FROZEN pipeline. Canonical one-shot
   seed: 20260710 (seed 20260709 belonged to an earlier, since-revised
   pipeline and was retired with it).
2. `--multiseed` mode — the one-shot point estimate still carries
   injection-schedule sampling variance, so ten further seeds
   (20260711–20260720) are PRE-COMMITTED in the file, all ten run with
   identical frozen code, and every run reported with no selection.
   Pooled: 381/420 = 0.907 [0.876, 0.931], precision 0.880. This is the
   generalization estimate to quote.

One-shot protocol: any future code change requires fresh seeds and
reporting all runs. A second independent iMAKS release remains the ideal
fix and is unavailable.

## Limitation 2 — Small n on the original set

14 events → recall 13/14 has 95% Wilson CI [0.685, 0.987]. The pooled
holdout (420 events) narrows the interval to ±0.03. Always report
intervals.

## Limitation 3 — The scoring rule is lenient; report the strictness curve

A GT event counts as COVERED on ANY nonzero overlap. The sweep
(`scoring_strictness.csv`): recall 0.929 → 0.857 at ≥10% overlap → 0.786
at ≥25% → 0.643 at ≥50%, and 0.571 with a 30-min latency cap. The drops
are the slow DRIFT events (GT-0002/0008/0012/0013): caught, but late and
partially. The holdout mirrors this honestly (pooled 0.907 → 0.795 at
≥25%).

## Limitation 4 — The plausibility filter carries the precision
## (transductivity RESOLVED by the calibration check)

Quarantine (violation rate > 0.5, unsupervised, no GT labels) removes 4
mis-extracted rules. Without it: FP 2 → 60, precision 0.867 → 0.178
(`sensitivity_analysis.csv`). Two former concerns, both now closed
empirically:

- **Cutoff tuning:** not finely tuned — quarantined rules sit at
  0.516–1.0 vs next-highest clean rule at 0.0125 (`violation_rates.csv`);
  any cutoff in that band is equivalent.
- **Transductivity:** the shipped filter computes rates over the full
  evaluation stream, which read as "using eval data". The calibration
  check (`calibration_filter_check.csv`, step5_sensitivity.py Part 3)
  recomputes the rates over only the PRE-REGISTERED first 8 h of the
  stream (10% — a commissioning period, deployable online before any
  scored anomaly). The quarantine set is IDENTICAL on both the original
  stream and the holdout injected stream (max per-rule rate difference
  <0.05 against a 0.5 cutoff), and the grid row
  "plausibility=calibration-8h" reproduces the shipped results exactly
  (13/14, FP 2, precision 0.867). The filter does not exploit
  anomaly-period data; full-stream computation is an implementation
  convenience, not an information advantage.

What remains true and must be stated: the filter is load-bearing —
extraction alone is precision-poor, and the pipeline's precision comes
from extraction plus this sanity layer.

## Limitation 5 — Metric definitions (state explicitly)

- Precision mixes units: TP counts GT windows covered, FP counts detected
  events matching no GT window (standard event-level formulation).
- The Phase 2 combined fraction (guide Table 2) — 19/22 = 86.4%, PASS at
  the ≥70% bar — mixes timeseries detection (13/14) with a purely
  STRUCTURAL maintenance check (6/8: a Maintenance node is COVERED if a
  MaintenanceRule merely governs its sensor). Report components separately.

## Limitation 6 — Sensitivity summary (12-config grid)

Recall is completely flat at 0.929 across the entire grid (on-delay, merge
window, plausibility cutoff, drift baseline ratio, drift minimum-evidence
guard) — no constant is load-bearing for recall. Precision varies only via
FP count (see Limitation 4; merge=30min would give precision 1.000 — the
shipped 15 min default was kept to avoid post-hoc tuning against GT).

## Context — baseline bracketing (`baseline_comparison.csv`)

| config | recall | precision | FPs |
|---|---|---|---|
| extracted-rules (pipeline) | 0.929 (13/14) | 0.867 | 2 |
| oracle GT thresholds | 0.857 (12/14) | 1.000 | 0 |
| rolling z-score z=2.5 | 0.929 (13/14) | 0.007 | 1862 |
| rolling z-score z=3.5 | 0.786 (11/14) | 0.060 | 172 |

The oracle (perfect threshold knowledge, bounds read from GT) structurally
cannot see GT-0004 — the STUCK event frozen inside the normal range — which
the pipeline catches via an extracted MaintenanceRule; the pipeline's
recall edge over the oracle comes from extracting *behavioral* rules, not
from better thresholds. The z-score baseline reaches comparable recall only
by raising 1,862 false events (precision 0.007). Knowledge extraction buys
recall AND precision simultaneously; that is the contribution claim.

## Limitation 7 — Dataset-guide vs nodes.csv ID discrepancy

The guide's §2.3.1 test-set table and `nodes.csv` disagree on the ID↔event
assignment within GT-0011..GT-0014 (e.g. the guide lists GT-0011 as the
WRH01 drift; nodes.csv assigns GT-0011 to the CHM01 out-of-range event).
The five events themselves are identical — only the ID labels are permuted.
All results here follow `nodes.csv`, which the guide names as the Phase 2
reference. Cite per-event results by sensor+type, not by bare GT ID.

## Limitation 8 — Single synthetic dataset

All results, including the holdout (same facility, same noise models, same
generative definitions), come from one synthetic dataset. Nothing here
estimates performance on real sensor data or other document styles.
