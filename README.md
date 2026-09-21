# Rule Extraction from Industrial SOPs Without a Target Schema

**Multi-Agent Extraction, Content-First Evaluation, and Knowledge-Graph Validation**

The full repository, including every script, corpus and scored artifact
referred to below, is public at
<https://github.com/dleflef/Rule-Extraction-from-Industrial-SOPs-Without-a-Target-Schema-Multi-Agent-Extraction-Content-First>.

This repository pulls operating rules out of industrial documents, loads them
into a knowledge graph, and then uses those rules, and nothing else, to flag
anomalies in plant telemetry.

You hand it an operating or control document. An SOP, a table of alarm
thresholds, a maintenance schedule, an access matrix. It gives back the rules
that document states, as structured records, binds those records to whatever
equipment the facility declares, and runs the surviving rules over sensor data.
None of it is preset in code. Which equipment ontology the document uses, which
fields its rules carry, which categories they fall into, and whether it is
written as prose or as a table or as a matrix all get worked out from the corpus.

That constraint is the whole point. Configure a pipeline for one corpus and you
can always make it score well on that corpus. The question I wanted answered was
how a pipeline that has been told nothing does on documents it has never seen,
from domains it was not built for.

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

`layers/layer_1/agent_1b_tools.py` converts PDFs to layout-aware Markdown with
IBM Docling, reads `.txt` files as they are, and then cleans up what comes back:
HTML decoding, escaped characters, broken ligatures, and the duplication that
vision bounding boxes leave in table cells. A cell can arrive as `"Inspect and
lubricate HIGH Inspect and lubricate HIGH"`. The de-duplication matches whole
tokens. A character-level rule looks fine on that example and then quietly eats
letters out of ordinary text.

`test_agent_1b.py` is the entry point. It converts everything in
`data/dataset/rules/` into `layers/layer_1/texts/`.

### Layer 2: multi-agent extraction

`layers/layer_2/step2_multi_agent_generic.py` is where most of the work happens.
It is a LangGraph graph with six stages.

1. **segment** (no LLM). Splits the document into blank-line-delimited blocks,
   packs the blocks into chunks, and numbers every content line. Provenance then
   belongs to the pipeline. No model has to remember to write it down, because
   an agent only ever gets asked to cite a line that is sitting in front of it. Headings are found by shape alone: short,
   isolated, no closing punctuation. Each one is carried into every chunk it
   covers.
2. **scout** (one LLM call per chunk, run in parallel). Emits one record per rule
   boundary that the document draws itself. That means a printed rule id where
   there is one, and otherwise a table row, a matrix cell, or a sentence that
   states a rule. Field names are copied from the document's own labels. There is
   no schema in the prompt and no worked example, since an example only ever
   demonstrates one document shape, and a model that has been shown one will
   reproduce it on documents that do not have it.
3. **induce** (one LLM call per corpus). The schema inducer. It never sees a
   document. What it sees is the inventory of field names and category labels the
   scouts actually produced, with occurrence counts and sample values.
4. **assemble** (no LLM). Groups records on the document's own record boundary,
   renames each field to its canonical name, and writes one row per rule. That
   boundary is either the printed identifier or the physical line the record came
   off, so granularity follows evidence printed in the document instead of a
   ground truth's column conventions.
5. **audit** (one LLM call per chunk, in parallel). Re-reads each assembled record
   against the numbered lines it cites, then drops or corrects whatever is not
   stated there. If a whole chunk comes back rejected, that counts as an audit
   failure and not as a finding. An auditor claiming every record is ungrounded
   is more likely to be broken than to be right.
6. **save** (no LLM). Writes the surviving records out as a rectangular CSV whose
   columns are whatever fields that run ended up with, alongside the long-format
   facts file and the induced schema as JSON.

Outputs land in `layers/layer_2/step2_results_generic/`:
`ext_multi_agent_generic_<corpus>_run<N>_<ts>.csv` for the records,
`facts_*.csv` for long-format provenance (one row per record, field and value),
and `audit_log_*.csv` when the audit sidecar is on.

Two comparison systems sit alongside it.

`baseline.py` is a deterministic regex parser. I wrote it by hand for the four
development documents, and it reads its entity inventory out of the seed graph.
It is per-corpus configuration by construction. It is there to measure what such
configuration buys and what it costs, not to transfer anywhere.

`step2_grid_search_extraction_en.py` is the earlier single-agent grid search over
models and prompting paradigms. It is kept around so the multi-agent pipeline can
be scored against it under one identical metric.

`llm_cache.py` keys responses on `SHA-256(model, messages)`. A warm cache replays
a run exactly, and any change to a prompt, a model or a document forces a real
call.

### Layer 3: evaluation

`layers/layer_3/step3_evaluation_generic_dynamic.py` reports one headline number,
`F1_content`. Putting three or four F1 variants side by side just invites quoting
whichever one is highest.

The evaluator assumes nothing about field names on either side. Exact category
matching cannot work here. Layer 2 discovers categories per document
(`EnvironmentalLimit`, `TriggeredResponse`) while each ground truth uses a fixed
taxonomy of its own (`OperationalRule`, `ThresholdRule`), and nothing ties the two
vocabularies together. So every row on both sides gets collapsed into one text
blob and compared as a whole, on two terms:

- semantic similarity of the two blobs, SBERT cosine;
- numeric agreement, scored in both directions. Every number in the GT row should
  turn up somewhere in the matched prediction, and every number the prediction
  carries should be justified by one in the GT row. A recall-only term charges
  nothing for surplus values, which makes it easy to game by padding a record
  with numbers that were never extracted.

Rows are paired by Hungarian assignment over the blended score. Two kinds of
column are dropped before any of that, and both get located structurally instead
of by a name whitelist: an id column, meaning whichever one holds a short
distinct value on almost every row, and a provenance column, meaning whichever
one holds the same value on all of them.

`step3_results/HOW_MATCHING_WORKS.md` writes out every comparison decision.
`step3_results/match_audit/<corpus>/` prints the exact strings that were compared
for every pair.

The remaining Layer 3 scripts test what `F1_content` cannot see:

| Script | Question it answers |
| --- | --- |
| `step3_validity_checks.py` | Does the metric measure extraction quality, or just a similarity floor? (null pairs, padding attacks, threshold curve, perturbations) |
| `step3_citation_validity.py` | Are the line citations resolvable, and do the cited lines contain what the record asserts? |
| `step3_field_verification.py` | Do numeric bounds sit in the right slot, and not merely somewhere in the record? |
| `step3_binding_check.py` | Slot binding across all four corpora, without inventing a field mapping |
| `step3_schema_stability.py` | Does the inducer produce a consistent corpus schema across repeated runs? |
| `step3_surplus_grounding.py` | Are unpaired surplus records fabrications, or facts the annotation has no row for? |
| `step3_audit_interventions.py` | What did the audit stage actually keep, drop, and correct? |
| `step3_audit_stats.py` | How decisive are the assignment's pairings (runner-up margins)? |
| `step3_calibration_sample.py` | Does the τ = 0.6 accept/reject boundary agree with a human? |
| `make_figures.py` | Renders every results figure from the committed CSVs, asserting each headline value before it draws |

`layers/oracle_baseline/oracle_gt_detect.py` gives the ceiling. Detection there is
driven by the human-written annotation itself, so the extraction's detection score
can be read against what perfect rules manage on the same telemetry.

### Layer 4: downstream use

`step5_core.py` holds the detection engine: four detectors (`threshold`, `stuck`,
`drift`, `sustained`), the plausibility guard, the alarm merge and the metrics.
Each is defined exactly once, so two experiments cannot drift apart through
duplicated logic. Its anti-leakage invariants hold at every entry point. Detection
reads `timestamp / sensor_id / value` and nothing else, and ground truth is opened
only by the scoring helpers, strictly afterwards.

Two harnesses consume it.

**`step4_detect_generic.py`** is an in-memory adapter from Layer 2's discovered
schema onto the detector stack. It is the only new code on purpose. Every detector
and every metric is imported unchanged, so any result here can be read as a
statement about the extraction itself. Sensors resolve by value against the
declared inventory, field names by mechanical normalisation, and the rule class by
what a record contains rather than what it is called. Keying on the discovered
label instead cost every threshold on the runs that chose `AlarmThreshold` over
`SensorThreshold`. Detector-ready rules dropped from 27 to 5 and detection F1 from
0.897 to 0.316, while `F1_content` sat at 0.719 ± 0.000 the entire time.

**`step4_graph_generic.py`** is the graph-mediated version, which is the question
the project actually poses. Every rule makes a round trip through Neo4j: load the
facility ABox, load the extracted rules as `(:Rule)` nodes with `GOVERNS` and
`APPLIES_TO` edges, validate by creating `GOVERNS_ABOX` only where the rule's
sensor matches a sensor the plant declares, then read the ACTIVE rules back out
with Cypher and detect on those. Records that cannot bind get loaded into the
graph as well, and that is deliberate. Load only the ones that already resolve and
`GOVERNS_ABOX` succeeds by construction, so validation reports 100% no matter what
the extraction did. Loading the others alongside lets the binding genuinely fail,
which is what turns the ACTIVE count into a real measurement.
Here 3 of 30 rule nodes name a sensor the plant does not declare, and they are
reported UNRESOLVED instead of being silently dropped. The leakage guard is
asserted while the run is going: the graph contains `AnomalyEvent` nodes, so
before detection starts the loaded rule set is checked for any ground-truth field
and the run aborts if one shows up.

`step4_detect_human.py` covers the records a telemetry detector cannot consume at
all, things like access authorisation, occupancy limits and acknowledgement
deadlines, against the human-domain logs. Each one has a null control that has to
collapse.

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

The three external corpora are held out. Each one was generated in a single pass,
document and ground truth together, by an external model. Generated once, then
left unread until it was scored. Their per-corpus `README.md` files record how.

---

## Setup

```bash
git clone https://github.com/dleflef/Rule-Extraction-from-Industrial-SOPs-Without-a-Target-Schema-Multi-Agent-Extraction-Content-First.git
cd Rule-Extraction-from-Industrial-SOPs-Without-a-Target-Schema-Multi-Agent-Extraction-Content-First

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` at the repo root and fill the values you need:

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
`--auditor-model`) default to `gemma4:31b`. I picked it because it replies
directly with no reasoning preamble, which keeps token budgets mapping cleanly
onto the structured prompts. One thing to watch: passing a `reasoning_effort` to
that model switches a thinking pass on, so the code sends none.

Neo4j is needed only for the graph round trip. Everything else runs without a
database, including all of Layer 3 and `step4_detect_generic.py`.

---

## Running it

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

Layer 2 flags worth knowing: `--runs N` (anything above 1 disables the cache
automatically), `--chunk-chars` (default 3500), `--max-concurrency` (default 4),
`--no-cache`, `--corpus-name`.

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
| `evaluate_grid.sh` | Re-scores the old grid runs under the same evaluator |

`evaluate_grid.sh` strips the grid's per-run bookkeeping columns (`model_name`,
`paradigm`, `level` and so on) before scoring. Left in, the evaluator would glue
`"gemma4:31b | cot_basic | 1 | False"` onto every prediction's content blob and
make the grid look worse than it is.

---

## Results

From `layers/step3_results/RESULTS.csv`, 5 runs per corpus:

| Corpus | GT rows | Records (mean) | F1_content | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Desalination (held out) | 26 | 27.2 ± 0.4 | **0.804 ± 0.018** | 0.787 | 0.823 |
| Development production line | 86 | 131.0 ± 0.0 | **0.719 ± 0.000** | 0.595 | 0.907 |
| Biogas (held out) | 47 | 28.0 ± 0.0 | **0.715 ± 0.023** | 0.957 | 0.570 |
| Sulfuric acid (held out) | 70 | 49.0 ± 1.0 | **0.649 ± 0.049** | 0.788 | 0.552 |

The precision and recall split has more to do with annotation granularity than
with extraction quality on its own. The development corpus extracts 131 records
against 86 annotated rows, so recall is high and precision low; biogas extracts 28
against 47 and goes the other way. `step3_surplus_grounding.py` is the check on
whether those surplus records are fabrications or unannotated facts.

Two configured systems sit beside it on the development corpus, scored by the
same evaluator. The grid search, handed that corpus's own thirteen field names,
reaches 0.949. The regex parser reaches 0.771. Both are above the pipeline's
0.719, and neither one moves: the grid's prompt names fields that only this
corpus has, and the parser needs a new function for every document you point it
at.

The parser is the one worth following downstream, because the two measures
disagree about it. Through the same detection stack it recovers 6 of the 14
anomalies against the pipeline's 13. A higher content score did not mean better
detection, and that gap is most of why Layer 4 exists.

Downstream, on the development corpus (`layers/layer_4/`), the extracted rules
find **all 14 of the plant's labelled anomalies** across 80 hours of telemetry,
211,200 readings from 22 sensors. Thirteen come from the rules on their own: ten
where a value crosses a threshold the rules carry, two from a drift condition,
one from a stuck condition.

The fourteenth is the one I care about. `GT-0009` is a packaging speed that
never leaves its extracted bounds, clearing the nearest by 0.0061 m/s at its
worst, so no single-sensor rule can fire on it. It comes back only through the
graph. Two records, read out of two different documents, name the same sensor
pair; that pair is stored as a `CORRELATES_WITH` edge; and when the sealing
current breaks its own extracted bound the detector follows the edge and asks
whether the coupled sensor is behaving oddly for itself. It is, and the window
it returns matches the recorded one to the second.

Give every rule the wrong sensor and those thirteen drop to four, so the
detections come from the rules and not from telemetry that would look odd
whatever you ran over it.

- In-memory. 27 of the 131 records are detector-ready: recall 0.929, precision
  0.867, **F1 0.897**.
- Graph round trip. Same rules through Neo4j, validated and read back out: 30
  rule nodes, 27 ACTIVE, **F1 0.897**. The `CORRELATES_WITH` edge takes recall to
  1.000 and **F1 0.933**.
- Scored leniently, which is worth saying out loud: a window counts as detected
  on any overlap. Rescoring the identical detections under stricter acceptance
  gives recall 0.714 at ≥25% window coverage and 0.500 at ≥50%. Both harnesses
  report those alongside.
- Human-domain checks. Access authorisation F1 1.000, occupancy limits F1 0.856,
  each against a null control.

Numbers move when the pipeline or the scoring parameters change. The CSVs under
`layers/step3_results/` are the source of truth, not this table.

---

## Where the artifacts live

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
exist to test that it does.

- **No per-corpus configuration.** No corpus overrides the id field, and no
  evaluation flag is tuned per domain. A generic evaluator that needs per-corpus
  hand-holding is not generic. This costs score in places and stays anyway: the
  desalination ground truth leaves `identifier` blank on 15 of its 26 rows, so
  detection falls through to `rule_text` and gives up 0.046 F1.
- **Nothing is keyed on a discovered column name.** Entities resolve by value
  against a declared inventory, and rule classes are decided by structure. A label
  whitelist over a discovered schema has already failed twice in this codebase,
  the same way both times.
- **Provenance is assigned by code, not remembered by a model.** Line numbering
  happens in `segment`, before any LLM sees anything.
- **Ground truth is read only to score.** Never to detect, never to extract, and
  the graph harness asserts it at run time.
- **Every claim traces to an artifact.** A number that cannot be reproduced from
  committed code and committed outputs does not get reported.

---

## What it comes down to

Point this at a facility's documents with no schema, no taxonomy and no worked
example, and the rules it writes down are good enough to run that plant's
alarms. On the development corpus they catch all 14 labelled anomalies: 13 from
the bounds and conditions the rules carry, and the last from a link between two
sensors that only the graph holds.

The regex parser scores higher than the pipeline on `F1_content` and finds six
of the fourteen. Agreeing with an annotation and producing rules that work are
not the same thing, and that is the reason Layer 4 is in this repository at all.
