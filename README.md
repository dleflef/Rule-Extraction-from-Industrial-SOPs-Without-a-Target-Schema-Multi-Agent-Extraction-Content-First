# Rule Extraction from Industrial SOPs Without a Target Schema

**Multi-Agent Extraction, Content-First**

Extracting machine-actionable operating rules from industrial documents, loading
them into a knowledge graph, and using them — and only them — to detect anomalies
in plant telemetry.

The system takes an operating or control document (an SOP, an alarm-threshold
table, a maintenance schedule, an access matrix), recovers the rules it states as
structured records, binds those records to the equipment a facility actually
declares, and streams the surviving rules over sensor data. Nothing about the
document is preset in code: which equipment ontology it uses, which fields its
rules carry, which categories they fall into, and whether it is written as prose,
a table, or a matrix are all discovered from the corpus itself.

That constraint is the point of the project. A pipeline configured per corpus can
always be made to score well on that corpus; the question here is what a pipeline
that is *told nothing* achieves on documents it has never seen, from domains it
was not built for.

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

### Layer 1 — document ingestion

`layers/layer_1/agent_1b_tools.py` converts PDFs to layout-aware Markdown with
IBM Docling and reads `.txt` files directly, then normalises the result: HTML
decoding, escaped characters, broken ligatures, and removal of vision
bounding-box duplication (a cell such as `"Inspect and lubricate HIGH Inspect and
lubricate HIGH"` is collapsed by matching **whole tokens**, never characters — a
character-level rule silently eats letters out of ordinary text).

`test_agent_1b.py` is the entry point: it converts every document in
`data/dataset/rules/` into `layers/layer_1/texts/`.

### Layer 2 — multi-agent extraction

`layers/layer_2/step2_multi_agent_generic.py` is the core of the system, built on
LangGraph. Five stages:

1. **segment** (no LLM) — splits each document into blank-line-delimited blocks,
   packs them into chunks, and **numbers every content line**. Provenance is
   therefore something the pipeline *knows* rather than something a model has to
   remember to write down: an agent is only ever asked to cite a line it is
   currently looking at. Section headings are detected by shape alone (short,
   isolated, unpunctuated) and carried into every chunk they cover.
2. **scout** (1 LLM call per chunk, run in parallel) — emits one record per rule
   the *document itself* delimits: a printed rule id if there is one, otherwise
   one table row, one matrix cell, or one rule-stating sentence. Field names are
   copied from the document's own labels. No schema and no worked example are
   supplied, because an example demonstrates one document shape and a model shown
   one reproduces it on documents that do not have it.
3. **induce** (1 LLM call per *corpus*) — the schema inducer. It never sees a
   document, only the inventory of field names and category labels the scouts
   actually produced, with occurrence counts and sample values.
4. **assemble** (no LLM) — groups records by the document's own record
   boundary, renames each field to its canonical name, and emits one row per
   rule. Because the boundary is the printed identifier or the physical line
   the record was read from, granularity is decided by evidence printed in
   the document and never by a ground truth's column conventions.
5. **audit** (1 LLM call per chunk, in parallel) — re-reads each assembled
   record against the numbered lines it cites and drops or corrects anything
   not actually stated there. A whole-chunk rejection is treated as an audit
   *failure* rather than a finding: an auditor claiming every record is
   ungrounded is likelier to be malfunctioning than to be right.

Outputs land in `layers/layer_2/step2_results_generic/`:
`ext_multi_agent_generic_<corpus>_run<N>_<ts>.csv` (the records),
`facts_*.csv` (long-format provenance: one row per record/field/value), and
`audit_log_*.csv` when the audit sidecar is enabled.

Two comparison systems live alongside it:

- `baseline.py` — a deterministic regex parser, hand-written for the four
  development documents and reading its entity inventory from the seed graph.
  It is per-corpus configuration by construction; the point is to measure what
  such configuration buys and costs, not to transfer.
- `step2_grid_search_extraction_en.py` — the earlier single-agent grid search
  over models × prompting paradigms, retained so the multi-agent pipeline can be
  scored against it under the identical metric.

`llm_cache.py` keys responses by `SHA-256(model, messages)`, so a warm cache
replays a run exactly and any prompt/model/document change forces a real call.

### Layer 3 — evaluation

`layers/layer_3/step3_evaluation_generic_dynamic.py` reports **one** headline
number, `F1_content`, because reporting several F1 variants side by side invites
quoting whichever is highest.

The evaluator assumes nothing about either side's field names. Category
exact-match is structurally unwinnable here — Layer 2 discovers categories per
document (`EnvironmentalLimit`, `TriggeredResponse`) while each ground truth uses
its own fixed taxonomy (`OperationalRule`, `ThresholdRule`), and nothing ties the
vocabularies together. So each row on both sides is collapsed into one text blob
and compared holistically:

- **semantic similarity** of the two blobs (SBERT cosine), and
- **numeric agreement**, scored *symmetrically* — every number in the GT row
  should appear somewhere in the matched prediction, and every number the
  prediction carries should be justified by one in the GT row. A recall-only term
  prices surplus values at zero and is exploitable by padding a record with
  numbers it never extracted.

Rows are paired by Hungarian assignment over the blended score. Only two column
kinds are dropped first, and both are located *structurally* rather than by a
name whitelist: an id column (whichever has a distinct, short value on almost
every row) and a provenance column (whichever holds the same value on every row).

`step3_results/HOW_MATCHING_WORKS.md` documents every comparison decision, and
`step3_results/match_audit/<corpus>/` prints the exact strings that were compared
for every pair.

The remaining Layer 3 scripts test the things `F1_content` cannot see:

| Script | Question it answers |
| --- | --- |
| `step3_validity_checks.py` | Does the metric measure extraction quality, or just a similarity floor? (null pairs, padding attacks, threshold curve, perturbations) |
| `step3_citation_validity.py` | Are the line citations resolvable, and do the cited lines contain what the record asserts? |
| `step3_field_verification.py` | Do numeric bounds sit in the *right slot*, not merely somewhere in the record? |
| `step3_binding_check.py` | Slot binding across all four corpora, without inventing a field mapping |
| `step3_schema_stability.py` | Does the inducer produce a consistent corpus schema across repeated runs? |
| `step3_surplus_grounding.py` | Are unpaired surplus records fabrications, or facts the annotation has no row for? |
| `step3_audit_interventions.py` | What did the audit stage actually keep, drop, and correct? |
| `step3_audit_stats.py` | How decisive are the assignment's pairings (runner-up margins)? |
| `step3_calibration_sample.py` | Does the τ = 0.6 accept/reject boundary agree with a human? |

`layers/oracle_baseline/oracle_gt_detect.py` is the ceiling: detection driven by
the human-written annotation itself, so the extraction's detection score can be
read against what *perfect* rules achieve on the same telemetry.

### Layer 4 — downstream use

`step5_core.py` holds the detection engine — four detectors (`threshold`,
`stuck`, `drift`, `sustained`), the plausibility guard, the alarm merge, and the
metrics — defined exactly once, so no two experiments can disagree through
duplicated logic. Its anti-leakage invariants hold at every entry point: only
`timestamp / sensor_id / value` are read during detection, and ground truth is
opened only by the scoring helpers, strictly afterwards.

Two harnesses consume it:

- **`step4_detect_generic.py`** — in-memory adapter from Layer 2's discovered
  schema to the detector stack. It is deliberately the *only* new code; every
  detector and metric is imported unchanged, so a result here is a statement
  about the extraction. Sensors are resolved **by value** against the declared
  inventory, field names by mechanical normalisation, and the rule class by what
  a record *contains* rather than what it is *called*. (Keying on the discovered
  label instead cost every threshold on runs that chose `AlarmThreshold` over
  `SensorThreshold`: 27 detector-ready rules became 5 and detection F1 fell from
  0.897 to 0.316, while `F1_content` reported 0.719 ± 0.000 throughout.)
- **`step4_graph_generic.py`** — the graph-mediated version, which is the
  question the project actually poses. Every rule makes a round trip through
  Neo4j: load the facility ABox, load the extracted rules as `(:Rule)` nodes with
  `GOVERNS` / `APPLIES_TO` edges, validate by creating `GOVERNS_ABOX` only where
  the rule's sensor matches a sensor the plant declares, then read the ACTIVE
  rules back out with Cypher and detect. Rules naming equipment that does not
  exist are reported UNRESOLVED, not silently dropped. The leakage guard is
  asserted at run time: the graph holds `AnomalyEvent` nodes, so before detection
  the loaded rule set is checked for any ground-truth field and the run aborts if
  one appears.

`step4_detect_human.py` exercises the records a telemetry detector *cannot*
consume — access authorisation, occupancy limits, acknowledgement deadlines —
against the human-domain logs, each with a null control that must collapse.

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

**The three external corpora are held out.** Each was generated in a single pass
(document + ground truth together) by an external model, generated once, and left
unread until it was scored. Their per-corpus `README.md` files record how.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` at the repo root:

```ini
# Neo4j — only needed for layers/layer_4/step4_graph_generic.py
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
`--auditor-model`) default to `gemma4:31b`, chosen because it replies directly
with no reasoning preamble, so token budgets map cleanly onto the structured
prompts. Note that passing a `reasoning_effort` to that model *turns a thinking
pass on* — the code deliberately sends none.

Neo4j is required **only** for the graph round trip. Everything else, including
all of Layer 3 and `step4_detect_generic.py`, runs without a database.

---

## Running it

### End to end

```bash
# Layer 1 — PDFs → layers/layer_1/texts/
python3 layers/layer_1/test_agent_1b.py

# Layers 2+3 — extract every corpus N times, then score each against its own GT
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

Useful Layer 2 flags: `--runs N` (N > 1 disables the cache automatically),
`--chunk-chars` (default 3500), `--max-concurrency` (default 4), `--no-cache`,
`--corpus-name`.

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
| `run_and_evaluate.sh` | The headline `RESULTS.csv` — all four corpora, 5 runs each |
| `run_desalination.sh` | Held-out desalination corpus: multi-agent + grid, both scored |
| `run_robustness_sweeps.sh` | Scoring-parameter sweeps (no LLM endpoint needed) |
| `run_audit_replication.sh` | Instrumented replication that logs every audit verdict |
| `evaluate_grid.sh` | Re-scores the old grid runs under the *same* evaluator |

`evaluate_grid.sh` strips the grid's per-run bookkeeping columns
(`model_name`, `paradigm`, `level`, …) before scoring — left in, the evaluator
would glue `"gemma4:31b | cot_basic | 1 | False"` onto every prediction's content
blob and make the grid look worse than it is.

---

## Results

`layers/step3_results/RESULTS.csv`, 5 runs per corpus:

| Corpus | GT rows | Records (mean) | F1_content | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Desalination (held out) | 26 | 27.2 ± 0.4 | **0.804 ± 0.018** | 0.787 | 0.823 |
| Development production line | 86 | 131.0 ± 0.0 | **0.719 ± 0.000** | 0.595 | 0.907 |
| Biogas (held out) | 47 | 28.0 ± 0.0 | **0.715 ± 0.023** | 0.957 | 0.570 |
| Sulfuric acid (held out) | 70 | 49.0 ± 1.0 | **0.649 ± 0.049** | 0.788 | 0.552 |

The precision/recall split tracks annotation granularity rather than extraction
quality alone: the development corpus extracts 131 records against 86 annotated
rows (high recall, low precision), while biogas extracts 28 against 47 (the
reverse). `step3_surplus_grounding.py` is the check on whether those surplus
records are fabrications or unannotated facts.

**Downstream, on the development corpus** (`layers/layer_4/`):

- In-memory detection — 27 of 131 records are detector-ready, producing
  recall 0.929 / precision 0.867 / **F1 0.897**, with the single miss being the
  CORRELATED event `GT-0009` that no single-signal detector can express
  (recall excluding it: 1.000).
- Graph round trip — the same rules loaded into Neo4j, validated, and read back
  out: 30 rule nodes → 27 ACTIVE, 100% ABox sensor coverage, **F1 0.897**; adding
  the `CORRELATES_WITH` edge recovers `GT-0009` for recall 1.000 / **F1 0.933**.
- Human-domain checks — access authorisation F1 1.000, occupancy limits F1 0.856,
  each against a null control.

Numbers move when the pipeline or the scoring parameters change; the CSVs in
`layers/step3_results/` are the source of truth, not this table.

---

## Where the artifacts live

```
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

These are the constraints the code is written to hold, and most of the Layer 3
scripts exist to test that it does.

- **No per-corpus configuration.** No corpus overrides the id field, and no
  evaluation flag is tuned per domain. A generic evaluator that needs per-corpus
  hand-holding is not generic — this is accepted even where it costs score (the
  desalination ground truth leaves `identifier` blank on 15 of 26 rows, so
  detection falls through to `rule_text`, costing 0.046 F1).
- **Nothing is keyed on a discovered column name.** Entities are resolved by
  value against a declared inventory; rule classes are decided by structure. A
  label whitelist over a discovered schema has already failed twice in this
  codebase, both times the same way.
- **Provenance is assigned by code, not remembered by a model.** Line numbering
  happens in `segment`, before any LLM sees anything.
- **Ground truth is read only to score.** Never to detect, never to extract, and
  the graph harness asserts this at run time rather than merely intending it.
- **Every claim traces to an artifact.** If a number cannot be reproduced from
  committed code and committed outputs, it does not get reported.
