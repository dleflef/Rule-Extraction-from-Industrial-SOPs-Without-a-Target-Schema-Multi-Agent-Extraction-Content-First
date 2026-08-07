# Schema-Agnostic Rule Extraction from Industrial SOPs

Master's thesis repository: a multi-agent LLM pipeline (LangGraph) that extracts
operating rules from industrial Standard Operating Procedure documents **without
being given a target schema**, plus a content-first evaluation metric
(F1_content) whose own validity is measured rather than assumed.

The written thesis is `main.tex` (compiled: `main.pdf`). Every figure it reports
traces to an artifact in this repository; the map is in section *Outputs* below.

---

## 1. What is evaluated

| Corpus | Role | Docs | GT rows | Ground truth |
|---|---|---|---|---|
| Production line (iMAKS) | development | 4 | 86 | `data/dataset/kg_seed/ground_truth.csv` |
| Biogas digestion | external test | 1 | 47 | `data/external_test_biogas/ground_truth_biogas.csv` |
| Cleanroom access | external test | 1 | 37 | `data/external_test_cleanroom/ground_truth_cleanroom.csv` |
| RO desalination | external test | 1 | 26 | `data/external_test_desalination/ground_truth_desalination.csv` |

The three external corpora were held out during development and are scored with
no per-corpus configuration of any kind. Every condition is run five times and
reported as mean ± standard deviation (the hosted endpoint is not
bit-reproducible under a fixed seed; see thesis §Methodology).

## 2. The two systems

- **Multi-agent pipeline (no schema supplied)** —
  `layers/layer_2/step2_multi_agent_generic.py`.
  Deterministic segmentation with line numbering → parallel *scout* agents that
  copy each document's own field labels → one *inducer* call that reconciles
  labels into a corpus-level schema without seeing a document → deterministic
  assembly → parallel grounding *audit*. All three agent roles run
  `gemma4:31b` (CLI-overridable per role).
- **Schema-specified grid (contrast condition)** —
  `layers/layer_2/step2_grid_search_extraction_en.py`.
  Ten prompting paradigms × two models (`gemma4:31b`, `nemotron-3-nano:30b`),
  prompt carries the development corpus's schema/taxonomy/identifier
  convention/worked examples. Applicable to the development corpus only.

## 3. Setup

```
pip install -r requirements.txt
```

`.env` in the project root:

```dotenv
OLLAMA_BASE_URL=<OpenAI-compatible endpoint>
OLLAMA_API_KEY=<key>
```

Nothing depends on the endpoint being hosted rather than local; any
OpenAI-compatible server works.

## 4. Reproducing the results

```zsh
# Extraction (5 repeats x 4 corpora, needs the LLM endpoint) + evaluation:
zsh layers/run_and_evaluate.sh 5

# Re-score existing predictions only (local, deterministic, no LLM calls):
zsh layers/run_and_evaluate.sh 5 --eval-only

# Metric validity: null case, perturbations, threshold sweep:
python3 layers/layer_3/step3_validity_checks.py

# Field-level bound-placement verification on the dev corpus (thesis §Field-Level Verification):
python3 layers/layer_3/step3_field_verification.py

# Aggregate audit-trail statistics (thesis §Reviewability):
python3 layers/layer_3/step3_audit_stats.py

# Slot-binding check on all four corpora, no field mapping (thesis §Binding Check):
python3 layers/layer_3/step3_binding_check.py

# Human-calibration instrument: blinded 60-pair annotation sheet + kappa scorer:
python3 layers/layer_3/step3_calibration_sample.py sample
python3 layers/layer_3/step3_calibration_sample.py score <filled_sheet.csv>

# Re-score the grid-search predictions with the current metric:
zsh layers/evaluate_grid.sh

# Architecture figure (generated from the pipeline module's own constants):
python3 layers/layer_2/visualize_pipeline.py
```

The evaluator can also be called directly on any GT/prediction pair:
`python3 layers/layer_3/step3_evaluation_generic_dynamic.py --gt <csv> --pred <csv> --audit`.
Every free parameter of the metric is a CLI flag.

## 5. Outputs (the artifacts behind the thesis figures)

| Thesis figure | Artifact |
|---|---|
| Main results table (4 corpora × 5 repeats) | `layers/step3_results/RESULTS.csv`, per-run detail in `runs_<corpus>_<ts>.csv` |
| Grid table rescored with F1_content | `layers/step3_results/GRID_RESCORED.csv` |
| Per-record match audit + worked example | `layers/step3_results/match_audit/<corpus>/` |
| F1 arithmetic per run, written out | `layers/step3_results/match_audit/<corpus>/f1_derivation_<corpus>.csv` |
| Metric validity (null / perturbations / threshold sweep) | `layers/step3_results/validity/` |
| Field-level verification (420/440 bound cells) | `layers/step3_results/field_verification_dev.csv` (+ `_mismatches.csv`) |
| Audit-trail aggregate stats (927 pairs, 73%/27%/18%) | `layers/step3_results/audit_trail_stats.csv` |
| Earlier run batch cited in §Results (variance discussion) | `logs/final_run.log` |
| Robustness: tag-number toggle, weight sweep, binding check | `layers/step3_results/robustness/` |
| Calibration instrument (blinded sheet + key) | `layers/step3_results/calibration/` |
| Raw pipeline predictions | `layers/layer_2/step2_results_generic/` |
| Raw grid predictions (100 runs) | `layers/layer_2/step2_results/` |

## 6. Compiling the thesis

```
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## 7. Repository history note

`layers/layer_4/` contains downstream tooling (Neo4j knowledge-graph population
and sensor-stream anomaly detection over extracted rules). It is not integrated
with the pipeline evaluated in the thesis and none of its output is reported
there; the thesis's §Limitations states this explicitly. `run_pipeline.py`,
`run_pipeline.bat` and `Preliminary/` belong to earlier iterations of the
project and are not entry points for anything the thesis reports.
