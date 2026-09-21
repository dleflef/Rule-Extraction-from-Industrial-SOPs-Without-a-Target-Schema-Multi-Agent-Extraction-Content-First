# Rule Extraction from Industrial SOPs Without a Target Schema

**Multi-Agent Extraction, Content-First Evaluation, and Knowledge-Graph Validation**

The full repository, including every script, corpus and scored artifact referred
to below, is available at
<https://github.com/dleflef/Rule-Extraction-from-Industrial-SOPs-Without-a-Target-Schema-Multi-Agent-Extraction-Content-First>.

This repository contains a prototype pipeline that attempts to extract operating
rules from industrial documents, load those rules into a knowledge graph, and
use the rules alone to flag anomalies in plant telemetry.

The intended input is an operating or control document: a standard operating
procedure, a table of alarm thresholds, a maintenance schedule, or an access
matrix. The pipeline returns the rules that the document appears to state, as
structured records, attempts to bind those records to the equipment the facility
declares, and applies the surviving rules to sensor data. No target schema is
fixed in the code. The equipment ontology a document uses, the fields its rules
carry, the categories those rules fall into, and whether the material is written
as prose, as a table or as a matrix are all inferred from the corpus itself.

This constraint is deliberate, though it is also a limitation. A pipeline
configured for a single corpus can usually be made to score well on that corpus,
which makes such results difficult to interpret. The question examined here is a
narrower one: how a pipeline given no prior information behaves on documents it
has not seen, drawn from domains it was not developed against. The evidence
below speaks only to the four corpora studied, and should be read with that
scope in mind.

---

## The pipeline

```
   PDF / TXT documents
          │
   ┌──────▼──────┐
   │  Layer 1    │  Docling layout-aware conversion → normalised .txt
   │  ingestion  │  layers/layer_1/
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │  Layer 2    │  segment → [scout]* → induce → assemble → [audit]* → save
   │  extraction │  LangGraph multi-agent, schema-agnostic
   └──────┬──────┘  layers/layer_2/
          │
   ┌──────▼──────┐
   │  Layer 3    │  F1_content (Hungarian assignment over blended
   │  evaluation │  semantic + numeric similarity), plus validity,
   └──────┬──────┘  robustness and provenance checks
          │         layers/layer_3/
   ┌──────▼──────┐
   │  Layer 4    │  rules → Neo4j → validate against the facility →
   │  downstream │  read back out → detect anomalies in telemetry
   └─────────────┘  layers/layer_4/
```

### Layer 1: document ingestion

`layers/layer_1/agent_1b_tools.py` converts PDFs to layout-aware Markdown using
IBM Docling, reads `.txt` files unchanged, and then applies a modest amount of
cleaning: HTML decoding, escaped characters, broken ligatures, and the
duplication that vision bounding boxes tend to leave in table cells. A cell may
arrive as `"Inspect and lubricate HIGH Inspect and lubricate HIGH"`. The
de-duplication operates on whole tokens. A character-level rule appears adequate
on that example but was found to remove letters from ordinary text elsewhere.

`test_agent_1b.py` is the entry point. It converts everything in
`data/dataset/rules/` into `layers/layer_1/texts/`.

### Layer 2: multi-agent extraction

`layers/layer_2/step2_multi_agent_generic.py` implements the greater part of the
processing. It is written as a LangGraph graph with six stages.

1. **segment** (no LLM). Splits the document into blank-line-delimited blocks,
   packs the blocks into chunks, and numbers every content line. Provenance is
   therefore assigned by the pipeline rather than by a model, since an agent is
   only ever asked to cite a line already placed in front of it. Headings are
   identified by shape alone: short, isolated, and without closing punctuation.
   Each heading is carried into every chunk it covers.
2. **scout** (one LLM call per chunk, run in parallel). Emits one record per rule
   boundary that the document itself draws. In practice this means a printed rule
   identifier where one exists, and otherwise a table row, a matrix cell, or a
   sentence stating a rule. Field names are copied from the document's own
   labels. The prompt contains neither a schema nor a worked example, on the
   reasoning that an example can only demonstrate one document shape and may
   encourage a model to reproduce that shape where it does not apply.
3. **induce** (one LLM call per corpus). The schema inducer, which never sees a
   document. Its input is the inventory of field names and category labels the
   scouts produced, together with occurrence counts and sample values.
4. **assemble** (no LLM). Groups records on the document's own record boundary,
   renames each field to its canonical name, and writes one row per rule. That
   boundary is either the printed identifier or the physical line the record came
   from, so granularity follows evidence printed in the document rather than the
   column conventions of a particular ground truth.
5. **audit** (one LLM call per chunk, in parallel). Re-reads each assembled record
   against the numbered lines it cites, then drops or corrects material that is
   not stated there. Where an entire chunk is rejected, this is recorded as an
   audit failure rather than as a finding. The assumption behind this is that an
   auditor rejecting an entire chunk is more plausibly in error than correct,
   although that assumption is not itself tested here.
6. **save** (no LLM). Writes the surviving records as a rectangular CSV whose
   columns are whatever fields that run produced, alongside the long-format facts
   file and the induced schema as JSON.

Outputs are written to `layers/layer_2/step2_results_generic/`:
`ext_multi_agent_generic_<corpus>_run<N>_<ts>.csv` for the records,
`facts_*.csv` for long-format provenance (one row per record, field and value),
and `audit_log_*.csv` when the audit sidecar is enabled.

Two comparison systems are provided alongside it.

`baseline.py` is a deterministic regex parser, written by hand for the four
development documents, which reads its entity inventory from the seed graph. It
is per-corpus configuration by construction. Its purpose is to indicate what such
configuration gains and what it costs, not to transfer to other material.

`step2_grid_search_extraction_en.py` is an earlier single-agent grid search over
models and prompting paradigms. It is retained so that the multi-agent pipeline
can be scored against it under a single identical metric.

`llm_cache.py` keys responses on `SHA-256(model, messages)`. A warm cache replays
a run exactly, and any change to a prompt, a model or a document forces a fresh
call.

### Layer 3: evaluation

`layers/layer_3/step3_evaluation_generic_dynamic.py` reports a single headline
number, `F1_content`. A single measure is reported in preference to several
variants, since reporting a family of related scores invites selective quotation
of the most favourable among them.

The evaluator assumes nothing about field names on either side. Exact category
matching is not available in this setting. Layer 2 discovers categories per
document (`EnvironmentalLimit`, `TriggeredResponse`), while each ground truth
uses a fixed taxonomy of its own (`OperationalRule`, `ThresholdRule`), and
nothing connects the two vocabularies. Every row on both sides is therefore
collapsed into a single text blob and compared as a whole, on two terms:

- semantic similarity of the two blobs, measured as SBERT cosine;
- numeric agreement, scored in both directions. Every number in the ground truth
  row should appear somewhere in the matched prediction, and every number the
  prediction carries should be justified by one in the ground truth row. A
  recall-only term would charge nothing for surplus values, which would make the
  metric easy to inflate by padding a record with numbers that were never
  extracted.

Rows are paired by Hungarian assignment over the blended score. Two kinds of
column are dropped beforehand, both located structurally rather than through a
name whitelist: an identifier column, meaning whichever column holds a short
distinct value on almost every row, and a provenance column, meaning whichever
column holds the same value on all of them.

`step3_results/HOW_MATCHING_WORKS.md` documents every comparison decision.
`step3_results/match_audit/<corpus>/` prints the exact strings compared for each
pair.

The remaining Layer 3 scripts examine what `F1_content` cannot show:

| Script | Question it addresses |
| --- | --- |
| `step3_validity_checks.py` | Does the metric measure extraction quality, or only a similarity floor? (null pairs, padding attacks, threshold curve, perturbations) |
| `step3_citation_validity.py` | Are the line citations resolvable, and do the cited lines contain what the record asserts? |
| `step3_field_verification.py` | Do numeric bounds sit in the correct slot, rather than merely somewhere in the record? |
| `step3_binding_check.py` | Slot binding across all four corpora, without introducing a field mapping |
| `step3_schema_stability.py` | Does the inducer produce a consistent corpus schema across repeated runs? |
| `step3_surplus_grounding.py` | Are unpaired surplus records fabrications, or facts for which the annotation has no row? |
| `step3_audit_interventions.py` | What did the audit stage keep, drop, and correct? |
| `step3_audit_stats.py` | How decisive are the assignment's pairings (runner-up margins)? |
| `step3_calibration_sample.py` | Does the τ = 0.6 accept/reject boundary agree with a human annotator? |
| `make_figures.py` | Renders every results figure from the committed CSVs, asserting each headline value before drawing |

`layers/oracle_baseline/oracle_gt_detect.py` provides a ceiling. Detection there
is driven by the human-written annotation itself, so the extraction's detection
score can be read against what perfect rules achieve on the same telemetry.

### Layer 4: downstream use

`step5_core.py` holds the detection engine: four detectors (`threshold`, `stuck`,
`drift`, `sustained`), the plausibility guard, the alarm merge and the metrics.
Each is defined once only, so that two experiments cannot diverge through
duplicated logic. Its anti-leakage invariants are enforced at every entry point.
Detection reads `timestamp / sensor_id / value` and nothing further, and ground
truth is opened only by the scoring helpers, strictly afterwards.

Two harnesses consume it.

**`step4_detect_generic.py`** is an in-memory adapter from Layer 2's discovered
schema onto the detector stack. It is, by design, the only new code introduced at
this stage. Every detector and every metric is imported unchanged, so results here
may reasonably be read as statements about the extraction itself. Sensors resolve by
value against the declared inventory, field names by mechanical normalisation,
and the rule class by what a record contains rather than by what it is called.
Keying on the discovered label instead was found to lose every threshold on runs
that chose `AlarmThreshold` over `SensorThreshold`. Detector-ready rules fell
from 27 to 5 and detection F1 from 0.897 to 0.316, while `F1_content` remained at
0.719 ± 0.000 throughout.

**`step4_graph_generic.py`** is the graph-mediated variant, which corresponds more
closely to the question the project poses. Every rule makes a round trip through
Neo4j: the facility ABox is loaded, the extracted rules are loaded as `(:Rule)`
nodes with `GOVERNS` and `APPLIES_TO` edges, validation creates `GOVERNS_ABOX`
only where the rule's sensor matches a sensor the plant declares, and the ACTIVE
rules are then read back out with Cypher and used for detection. Records that
cannot bind are loaded into the graph as well, which is deliberate. Were only the
records that already resolve to be loaded, `GOVERNS_ABOX` would succeed by
construction and validation would report 100% regardless of extraction quality.
Loading the remainder alongside allows the binding to fail where it should, which
is what makes the ACTIVE count informative. In these runs 3 of 30 rule nodes name
a sensor the plant does not declare, and they are reported UNRESOLVED rather than
silently dropped. The leakage guard is asserted during the run: the graph contains
`AnomalyEvent` nodes, so before detection begins the loaded rule set is checked
for any ground-truth field and the run aborts if one is present.

`step4_detect_human.py` covers records that a telemetry detector cannot consume at
all, such as access authorisation, occupancy limits and acknowledgement deadlines,
evaluated against the human-domain logs. Each is accompanied by a null control
that is expected to collapse.

---

## Repository layout

```
data/
  dataset/
    rules/         4 development SOP PDFs (operating, alarms, maintenance, access)
    kg_seed/       facility ABox + ground truth (nodes, edges, nodes_factory, ground_truth)
    sensors/       telemetry (raw + annotated) and MQTT payloads
    human/         access events, occupancy, alarm responses, person registry
    datasheets/    sensor datasheet PDFs
    csi/           1100 CSI activity files, 44 subjects (not used by the current pipeline)
  external_test_biogas/          held-out corpus + ground truth + README
  external_test_desalination/    "
  external_test_sulfuric_acid/   "
layers/
  layer_1/  ingestion            layer_3/  evaluation + validity/robustness
  layer_2/  extraction           layer_4/  detection, graph round trip
  oracle_baseline/               perfect-rule detection ceiling
  step3_results/                 all scored artifacts (see below)
  *.sh                           reproduction drivers
Preliminary/                     early exploratory notebooks and demos
```

The three external corpora are held out. Each was generated in a single pass,
document and ground truth together, by an external model, generated once and left
unread until scoring. Their per-corpus `README.md` files record the procedure.

---

## Setup

```bash
git clone https://github.com/dleflef/Rule-Extraction-from-Industrial-SOPs-Without-a-Target-Schema-Multi-Agent-Extraction-Content-First.git
cd Rule-Extraction-from-Industrial-SOPs-Without-a-Target-Schema-Multi-Agent-Extraction-Content-First

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` at the repository root and fill in the values
required:

```bash
cp .env.example .env
```

The production pipeline uses the following subset:

```ini
# Neo4j: only needed for layers/layer_4/step4_graph_generic.py
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j

# LLM backend (OpenAI-compatible endpoint)
OLLAMA_BASE_URL=...
OLLAMA_API_KEY=...
OLLAMA_EXTRACTION_MODEL=gemma4:31b
OLLAMA_SMALL_MODEL=...
OLLAMA_JUDGE_MODEL=...
```

All three Layer 2 agent roles (`--scout-model`, `--inducer-model`,
`--auditor-model`) default to `gemma4:31b`. It was selected because it replies
directly without a reasoning preamble, which keeps token budgets mapping cleanly
onto the structured prompts. A practical consideration is that passing a
`reasoning_effort` to this model enables a thinking pass, so the code sends
none.

Neo4j is required only for the graph round trip. Everything else runs without a
database, including all of Layer 3 and `step4_detect_generic.py`.

---

## Running the pipeline

### End to end

```bash
# Layer 1: PDFs → layers/layer_1/texts/
python3 layers/layer_1/test_agent_1b.py

# Layers 2+3: extract every corpus N times, then score each against its own GT
zsh layers/run_and_evaluate.sh 5
zsh layers/run_and_evaluate.sh 5 --eval-only    # re-score without re-running the LLM
```

### A single corpus

```bash
cd layers/layer_2
python3 step2_multi_agent_generic.py --input-dir ../../data/external_test_biogas --runs 5

python3 ../layer_3/step3_evaluation_generic_dynamic.py \
    --pred-dir step2_results_generic \
    --gt ../../data/external_test_biogas/ground_truth_biogas.csv \
    --aggregate --report ../step3_results/biogas.csv
```

The principal Layer 2 flags are `--runs N` (any value above 1 disables the cache
automatically), `--chunk-chars` (default 3500), `--max-concurrency` (default 4),
`--no-cache` and `--corpus-name`.

### Downstream detection

```bash
python3 layers/layer_4/step4_detect_generic.py --rules <extraction.csv>   # no DB
python3 layers/layer_4/step4_graph_generic.py --all-dev-runs              # via Neo4j
python3 layers/layer_4/step4_detect_human.py                              # human-domain logs
python3 layers/oracle_baseline/oracle_gt_detect.py                        # perfect-rule ceiling
```

### Reproduction drivers

| Script | What it reproduces |
| --- | --- |
| `run_and_evaluate.sh` | The headline `RESULTS.csv`: all four corpora, 5 runs each |
| `run_desalination.sh` | Held-out desalination corpus, multi-agent + grid, both scored |
| `run_robustness_sweeps.sh` | Scoring-parameter sweeps (no LLM endpoint needed) |
| `run_audit_replication.sh` | Instrumented replication that logs every audit verdict |
| `evaluate_grid.sh` | Re-scores the earlier grid runs under the same evaluator |

`evaluate_grid.sh` strips the grid's per-run bookkeeping columns (`model_name`,
`paradigm`, `level` and so on) before scoring. If they were left in, the evaluator
would append `"gemma4:31b | cot_basic | 1 | False"` to every prediction's content
blob and understate the grid's performance.

---

## Results

From `layers/step3_results/RESULTS.csv`, 5 runs per corpus:

| Corpus | GT rows | Records (mean) | F1_content | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Desalination (held out) | 26 | 27.2 ± 0.4 | **0.804 ± 0.018** | 0.787 | 0.823 |
| Development production line | 86 | 131.0 ± 0.0 | **0.719 ± 0.000** | 0.595 | 0.907 |
| Biogas (held out) | 47 | 28.0 ± 0.0 | **0.715 ± 0.023** | 0.957 | 0.570 |
| Sulfuric acid (held out) | 70 | 49.0 ± 1.0 | **0.649 ± 0.049** | 0.788 | 0.552 |

The split between precision and recall appears to reflect annotation granularity
more than extraction quality on its own. The development corpus extracts 131
records against 86 annotated rows, giving high recall and low precision; biogas
extracts 28 against 47 and shows the opposite pattern. `step3_surplus_grounding.py`
is the check on whether those surplus records are fabrications or unannotated
facts.

Two configured systems sit beside the pipeline on the development corpus, scored
by the same evaluator. The grid search, supplied with that corpus's own thirteen
field names, reaches 0.949. The regex parser reaches 0.771. Both exceed the
pipeline's 0.719, and neither transfers: the grid's prompt names fields specific
to this corpus, and the parser requires a new function for each document it is
pointed at.

The parser merits closer attention downstream, since the two measures disagree
about it. Through the same detection stack it recovers 6 of the 14 anomalies,
against the pipeline's 13. A higher content score did not correspond to better
detection in this case, and that divergence is much of the reason Layer 4 was
added.

Downstream, on the development corpus (`layers/layer_4/`), the extracted rules
identify **all 14 of the plant's labelled anomalies** across 80 hours of
telemetry, comprising 211,200 readings from 22 sensors. Thirteen follow from the
rules alone: ten where a value crosses a threshold the rules carry, two from a
drift condition, and one from a stuck condition.

The fourteenth case is of particular interest. `GT-0009` is a packaging speed that
never leaves its extracted bounds, clearing the nearest by 0.0061 m/s at its
worst, so no single-sensor rule can fire on it. It is recovered only through the
graph. Two records, read from two different documents, name the same sensor pair;
that pair is stored as a `CORRELATES_WITH` edge; and when the sealing current
breaks its own extracted bound, the detector follows the edge and asks whether the
coupled sensor is behaving unusually for itself. It is, and the window returned
matches the recorded one to the second.

Assigning every rule the wrong sensor reduces those thirteen to four, which
suggests the detections derive from the rules rather than from telemetry that
would appear anomalous under any procedure.

- In-memory. 27 of the 131 records are detector-ready: recall 0.929, precision
  0.867, **F1 0.897**.
- Graph round trip. The same rules through Neo4j, validated and read back out: 30
  rule nodes, 27 ACTIVE, **F1 0.897**. The `CORRELATES_WITH` edge raises recall to
  1.000 and gives **F1 0.933**.
- The scoring is lenient, and this should be taken into account when reading the
  figures above: a window counts as detected on any overlap. Rescoring the
  identical detections under stricter acceptance gives recall 0.714 at ≥25% window
  coverage and 0.500 at ≥50%. Both harnesses report these figures alongside the
  headline values.
- Human-domain checks. Access authorisation F1 1.000, occupancy limits F1 0.856,
  each against a null control.

These numbers change when the pipeline or the scoring parameters change. The CSVs
under `layers/step3_results/` are the authoritative record, not this table.

---

## Location of the artifacts

```
layers/layer_3/
  figures/                    results figures, regenerated by make_figures.py
layers/step3_results/
  RESULTS.csv                 headline table (above)
  GRID_RESCORED.csv           the grid search under the same metric
  HOW_MATCHING_WORKS.md       every comparison decision, explained
  match_audit/<corpus>/       every pair, with the exact strings compared
  validity/                   null pairs, padding attacks, threshold curve, perturbations
  robustness/                 parameter sweeps, sensitivity grids, citation validity,
                              slot agreement, surplus grounding + its null control
  calibration/                human-annotation sheet, key, and instructions
  runs_* / summary_*          per-run and per-corpus scoring output
layers/layer_2/
  step2_results_generic/      multi-agent records, facts_* provenance, audit logs
  step2_results/              grid-search runs
  baseline_results/           regex baseline
layers/layer_4/
  detection_generic_results/  in-memory detection, human-domain checks, null controls
  graph_generic_results/      graph round-trip detection
layers/oracle_baseline/       oracle_coverage.csv, oracle_strictness.csv
```

---

## Design commitments

These are the constraints the code is written to hold. Most of the Layer 3 scripts
exist to test whether it does.

- **No per-corpus configuration.** No corpus overrides the identifier field, and
  no evaluation flag is tuned per domain. An evaluator that required per-corpus
  adjustment could not reasonably be described as generic. The constraint costs
  score in places and is retained regardless: the desalination ground truth leaves
  `identifier` blank on 15 of its 26 rows, so detection falls through to
  `rule_text` and gives up 0.046 F1.
- **Nothing is keyed on a discovered column name.** Entities resolve by value
  against a declared inventory, and rule classes are determined by structure. A
  label whitelist over a discovered schema proved unreliable on two separate
  occasions during development, in the same manner each time.
- **Provenance is assigned by code, not remembered by a model.** Line numbering
  occurs in `segment`, before any LLM sees the material.
- **Ground truth is read only for scoring.** Never for detection, never for
  extraction, and the graph harness asserts this at run time.
- **Every reported number traces to an artifact.** A figure that cannot be
  reproduced from committed code and committed outputs is not reported.

---

## Summary

Given a facility's documents with no schema, no taxonomy and no worked example,
the pipeline produced rules that were sufficient to reproduce the labelled alarms
of the plant in the experiments reported here. On the development corpus the
extracted rules account for all 14 labelled anomalies: 13 from the bounds and
conditions the rules carry, and the remaining one from a link between two sensors
held only in the graph.

The regex parser scores higher than the pipeline on `F1_content` while recovering
six of the fourteen anomalies. Agreement with an annotation and the production of
rules that function downstream are therefore not equivalent, which is the reason
Layer 4 forms part of this repository. These observations rest on four corpora and
a single facility, and further evaluation would be needed before generalising
from them.
