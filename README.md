# Agentic Knowledge Graph for Digital Twins

A Multi-Agent System (MAS) that automatically constructs and validates Knowledge Graphs from industrial Standard Operating Procedure (SOP) documents, then uses those graphs as the detection backbone for a Digital Twin anomaly-detection pipeline.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Environment Setup](#3-environment-setup)
4. [Full Pipeline Schema](#4-full-pipeline-schema)
5. [Step-by-Step Execution](#5-step-by-step-execution)
   - [Layer 1 — Document Parsing](#layer-1--document-parsing)
   - [Layer 2a — Grid Search Extraction](#layer-2a--grid-search-extraction)
   - [Layer 2b — Multi-Agent Extraction + Evaluation](#layer-2b--multi-agent-extraction--evaluation)
   - [Layer 3 — Consensus & Evaluation (embedded in 2b)](#layer-3--consensus--evaluation)
   - [Layer 4a — Knowledge Graph Population](#layer-4a--knowledge-graph-population)
6. [Default Parameters Reference](#6-default-parameters-reference)
7. [Key Outputs & Metrics](#7-key-outputs--metrics)
8. [Reproducibility Notes](#8-reproducibility-notes)

---

## 1. Project Overview

The system answers: **"Can a multi-agent LLM pipeline extract industrial operating rules from SOP documents reliably enough that those rules detect real sensor anomalies?"**

The pipeline has four layers:

| Layer | Responsibility |
|---|---|
| 1 | Parse PDF/TXT SOPs → plain text chunks |
| 2 | Extract rules from text (grid-search over paradigms OR multi-agent pipeline) |
| 3 | Evaluate extraction quality; build consensus rule set |
| 4 | Load consensus rules into Neo4j; detect anomalies in raw sensor data; report |

The Layer 4 detection is **blind**: the knowledge graph contains only LLM-extracted rules; ground-truth data is used only as a post-hoc answer key.

---

## 2. Repository Layout

```
Agentic_KnowledgeGraph_DigitalTwins/
├── .env                              # LLM endpoints + Neo4j credentials (see §3)
├── requirements_step2.txt            # Python dependencies
├── data/
│   ├── dataset/
│   │   ├── kg_seed/
│   │   │   ├── ground_truth.csv      # Answer key (146 rules); used only for eval
│   │   │   ├── nodes.csv             # Ground-truth KG nodes (answer key for Layer 4)
│   │   │   ├── edges.csv             # Ground-truth KG edges (answer key for Layer 4)
│   │   │   └── nodes_factory.csv     # ABox seed nodes (input to graph_informed paradigm)
│   │   └── sensors/
│   │       └── timeseries_raw.csv    # Raw sensor time-series (input to step5)
│   └── seed_rules.zip                # Zipped SOP documents (input to Layer 1)
└── layers/
    ├── main.py                       # Layer 0 ReAct demo (standalone)
    ├── texts/                        # OUTPUT of Layer 1 (.txt files)
    ├── state/                        # Checkpoints (experiment_registry.json, metadata CSVs)
    ├── layer_1/
    │   ├── agent_1b_tools.py         # PDF/TXT extraction utilities (Docling)
    │   ├── agent_1b_state.py         # LangGraph TypedDict state schema
    │   └── agent_1b.py              # LangGraph workflow: START → parse → END
    ├── layer_2/
    │   ├── step2_grid_search_extraction_en.py   # Grid search: models × paradigms
    │   ├── step2_multi_agent_baseline.py        # Multi-agent pipeline (LangGraph)
    │   ├── step2_multi_run_eval.py              # Multi-run evaluation + consensus
    │   ├── baseline.py                          # Deterministic regex baseline (B0)
    │   ├── visualize_pipeline.py                # Renders pipeline diagram as PNG
    │   └── step2_results/                       # OUTPUT: extraction CSVs
    ├── layer_3/
    │   └── step3_evaluation_rules.py            # iMAKS benchmark (SBERT + exact-match)
    ├── step3_results/                            # OUTPUT: eval summaries, consensus rules
    ├── layer_4/
    │   └── step4_populate.py          # Load consensus rules → Neo4j
    └── test_agent_1b.py              # Batch-convert zipped SOPs → texts/
```

---

## 3. Environment Setup

### Install dependencies

```bash
pip install -r requirements_step2.txt
pip install docling sentence-transformers neo4j python-dotenv langchain langgraph scipy
```

### Configure `.env`

Create `.env` in the project root:

```dotenv
# Neo4j instance (Layer 4)
NEO4J_URI=bolt://<host>:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j

# LLM endpoint — Ollama (remote) or LM Studio (local)
OLLAMA_BASE_URL=https://<ollama-host>/v1
OLLAMA_API_KEY=<key>

LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODELS=qwen/qwen3-4b-2507
MODEL_ORCHESTRATOR=qwen/qwen3-4b-2507
```

All scripts load this file via `python-dotenv`; no other credential files are needed.

---

## 4. Full Pipeline Schema

```
data/seed_rules.zip
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 1 — Document Parsing                             │
│  test_agent_1b.py                                       │
│    └─► agent_1b.py (LangGraph)                         │
│          └─► agent_1b_tools.py (Docling PDF parser)    │
│  Output: layers/texts/*.txt                             │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌─────────────────────┐   ┌──────────────────────────────┐
│  LAYER 2a           │   │  LAYER 2b                    │
│  Grid Search        │   │  Multi-Agent Pipeline        │
│                     │   │                              │
│  step2_grid_search  │   │  step2_multi_run_eval.py     │
│  _extraction_en.py  │   │    └─► step2_multi_agent_    │
│                     │   │        baseline.py           │
│  10 paradigms ×     │   │        (LangGraph StateGraph)│
│  5 models × seeds   │   │        coordinator           │
│                     │   │        ├─ extractor-A (narr) │
│  Output:            │   │        ├─ extractor-B (tab)  │
│  step2_results/     │   │        └─ extractor-C (mat)  │
│  ext_*.csv          │   │        merge → validate      │
└─────────────────────┘   │        → judge → normalize   │
                          │                              │
                          │  Output:                     │
                          │  step2_results/              │
                          │  ext_multi_agent_run*.csv    │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │  LAYER 3 — Evaluation        │
                          │  (embedded in step2_multi_   │
                          │   run_eval.py)               │
                          │                              │
                          │  step3_evaluation_rules.py   │
                          │  SBERT text fields           │
                          │  Exact-match categoricals    │
                          │  Tolerance numeric fields    │
                          │                              │
                          │  ─ Computes F1_strict,       │
                          │    F1_content, IAAS,         │
                          │    Fleiss κ                  │
                          │  ─ Consensus = rules in      │
                          │    ≥70% of runs              │
                          │                              │
                          │  Output:                     │
                          │  step3_results/              │
                          │  consensus_rules.csv  ◄──────│── CRITICAL ARTIFACT
                          │  iaas_report.json            │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │  LAYER 4a — KG Population    │
                          │  step4_populate.py           │
                          │                              │
                          │  consensus_rules.csv →       │
                          │  parse conditions →          │
                          │  Neo4j nodes & edges         │
                          │                              │
                          │  :Station, :Sensor, :Rule    │
                          │  HAS_SENSOR, GOVERNS,        │
                          │  APPLIES_TO, IMPLICATES      │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
```

---

## 5. Step-by-Step Execution

All commands are run from the `layers/` directory unless noted.

### Layer 1 — Document Parsing

Converts zipped SOP documents (PDF/TXT) to plain-text chunks.

```bash
cd layers
python test_agent_1b.py
```

**Input:** `../data/seed_rules.zip`  
**Output:** `layers/texts/*.txt`

Default constants in `test_agent_1b.py`:

| Parameter | Value |
|---|---|
| `DEFAULT_ZIP` | `data/seed_rules.zip` |
| `TEXTS_DIR` | `texts` |
| `_MIN_CHUNK_CHARS` | `200` |
| `_MAX_CHUNK_CHARS` | `2000` |

---

### Layer 2a — Grid Search Extraction

Runs all combinations of model × paradigm × seed and scores each with F1 against ground truth.

```bash
# Full grid (default)
python layer_2/step2_grid_search_extraction_en.py

# Subset: specific models
python layer_2/step2_grid_search_extraction_en.py --models qwen/qwen3-4b-2507 gemma3:12b

# Subset: specific paradigms
python layer_2/step2_grid_search_extraction_en.py --paradigms naive cot_basic

# Force re-run (ignore checkpoint registry)
python layer_2/step2_grid_search_extraction_en.py --force

# Multiple seeds
python layer_2/step2_grid_search_extraction_en.py --seeds 42 123 7
```

**Input:** `layers/texts/*.txt`, `data/dataset/kg_seed/ground_truth.csv`  
**Output:** `layers/layer_2/step2_results/ext_<model>_<paradigm>_s<seed>_run<N>.csv`

Default constants:

| Parameter | Value | Location |
|---|---|---|
| Models | `ministral-3:14b`, `gemma3:12b`, `ministral-3:8b`, `qwen/qwen3-4b-2507`, `gpt-oss:20b` | line ~50 |
| Paradigms | 10: `naive`, `few_shot_static`, `graph_informed`, `cot_basic`, `cot_structured`, `pre_act`, `self_consistency`, `reflexion`, `reflexion_guided`, `react_abox` | line ~60 |
| `N_RUNS` | `1` | line 97 |
| `SEED` | `42` | line 98 |
| `SELF_CONS_RUNS` | `3` | line 106 |
| `SELF_CONS_TEMP` | `0.3` | line 106 |
| `MAJORITY_THRESH` | `2` | line ~110 |
| `ENSEMBLE_AGREE_THR` | `0.55` | line ~112 |
| `MAX_REACT_TURNS` | `5` | line ~115 |
| `LLM_TIMEOUT_SEC` | `600` | line ~120 |
| `LLM_NUM_CTX` | `16384` | line ~122 |
| `MAX_OUTPUT_TOKENS` | `16384` | line ~123 |
| `MAX_RETRIES` | `3` | line ~125 |
| `RETRY_BASE_DELAY` | `15.0` s | line ~126 |
| `SOP_TEXT_LIMIT` | `8000` chars | line ~130 |

Completed runs are checkpointed in `layers/state/experiment_registry.json`; re-running skips completed entries automatically.

---

### Layer 2b — Multi-Agent Extraction + Evaluation

Runs the production multi-agent pipeline N times, then computes cross-run agreement and consensus.

```bash
# Default: 20 runs at temperature 0.5
python layer_2/step2_multi_run_eval.py

# Custom run count
python layer_2/step2_multi_run_eval.py --runs 5

# Custom temperature
python layer_2/step2_multi_run_eval.py --temperature 0.5

# Recompute stats from existing CSVs (skip new LLM calls)
python layer_2/step2_multi_run_eval.py --skip-runs
```

**Input:** `layers/texts/*.txt`, `data/dataset/kg_seed/nodes_factory.csv` (ABox seed)  
**Output:** `layers/layer_2/step2_results/multi_run/ext_multi_agent_run*.csv`

Multi-agent pipeline model assignments (defaults in `step2_multi_agent_baseline.py`):

| Agent | Model | Role |
|---|---|---|
| Coordinator | `qwen/qwen3-4b-2507` | Document type classification |
| Extractor-A | `gemma3:12b` | Narrative / mixed SOPs |
| Extractor-B | `ministral-3:8b` | Tabular threshold rules |
| Extractor-C | `gemma3:12b` | Matrix / access rules |
| Validator | `ministral-3:14b` | Gap-fill verification |
| Judge | `ministral-3:14b` | Remaining-gap detection |
| Normalizer | `qwen/qwen3-4b-2507` | ruleId canonicalisation |

Other pipeline constants:

| Parameter | Value |
|---|---|
| `_AGREEMENT_THRESHOLD` | `0.75` (field match ratio for duplicate collapse) |
| `MULTI_RUN_TEMPERATURE` | `0.5` |
| Physics check | `critLo ≤ warnLo ≤ warnHi ≤ critHi` enforced |

Ablation variants run automatically alongside the full pipeline:

| Variant | What changes |
|---|---|
| `abl_noCM` | Coordinator replaced by keyword regex (no LLM) |
| `abl_noValidator` | Validator step skipped |
| `abl_noJudge` | Judge step skipped |

---

### Layer 3 — Consensus & Evaluation

This layer runs **inside** `step2_multi_run_eval.py` via `step3_evaluation_rules.py`. No separate invocation is needed.

**Output files written to `layers/step3_results/`:**

| File | Contents |
|---|---|
| `multi_run_stats.csv` | F1 mean ± std, 95 % CI across N runs |
| `iaas_report.json` | IAAS score, Fleiss κ, consensus rule set |
| `consensus_rules.csv` | Rules present in ≥ 70 % of runs — **fed to Layer 4** |
| `evaluation_summary.csv` | Per-run F1_strict, F1_content, precision, recall |

Evaluation constants in `step3_evaluation_rules.py`:

| Parameter | Value |
|---|---|
| SBERT model | `all-MiniLM-L6-v2` |
| Consensus threshold | `CONSENSUS_FRAC = 0.70` |
| Hallucination proxy | `HALLUCINATION_FRAC = 0.30` |
| Similarity threshold (IAAS) | `SIMILARITY_THRESHOLD = 0.90` |

Field weights used for F1_content (ThresholdRule example):

| Field | Weight | Scoring |
|---|---|---|
| `class` | 1.5 | exact match |
| `station` | 2.0 | exact match |
| `sensor` | 2.5 | exact match |
| `sensorType` | 1.0 | exact match |
| `severity` | 0.5 | exact match |
| `unit` | 1.0 | exact match |
| `critHi` / `critLo` | 3.0 each | numeric tolerance |
| `warnHi` / `warnLo` | 1.5 each | numeric tolerance |
| `condition` | 0.5 | SBERT cosine |
| `action` | 0.5 | SBERT cosine |

---

### Layer 4a — Knowledge Graph Population

Loads consensus rules into Neo4j. Run after `step3_results/consensus_rules.csv` exists.

```bash
# Wipe graph and reload (default)
python layer_4/step4_populate.py

# Preserve existing nodes, add new
python layer_4/step4_populate.py --no-clear
```

**Input:** `layers/step3_results/consensus_rules.csv`  
**Output:** Neo4j database with nodes and relationships

Graph schema created:

```
(:Station {stationId})
    -[:HAS_SENSOR]->
(:Sensor {sensorId, sensorType, unit})
    <-[:APPLIES_TO]-
(:Rule {
    ruleId, class, condition, action, severity,
    critHi, warnHi, warnLo, critLo, unit,
    stuckSamples,
    driftDelta, driftMinutes,
    susOp, susValue, susMinutes,
    corrSourceSensor, corrTargetSensor,
    sourceFile
})
    -[:GOVERNS]->
(:Station)
    and [:IMPLICATES]-> (:Sensor)
```

Condition parsing extracts machine-usable parameters from free-text:

| Free-text condition | Parsed fields |
|---|---|
| `TEN STUCK >10 samples` | `stuckSamples=10` |
| `CUR DRIFT >1.0 A for 30 min` | `driftDelta=1.0, driftMinutes=30` |
| `PRS <3.3 bar for >15 min` | `susOp=<, susValue=3.3, susMinutes=15` |

---

## 6. Default Parameters Reference

Quick lookup for all parameters that affect numerical results:

| Parameter | Value | File | Note |
|---|---|---|---|
| Seed | `42` | `step2_grid_search_extraction_en.py:98` | Use `--seeds` to override |
| Grid N_RUNS | `1` | `step2_grid_search_extraction_en.py:97` | Set 10 for final eval |
| Multi-run count | `20` | `step2_multi_run_eval.py` | Use `--runs` to override |
| Multi-run temperature | `0.5` | `step2_multi_run_eval.py` | Use `--temperature` to override |
| Self-consistency runs | `3` | `step2_grid_search_extraction_en.py:106` | |
| Self-consistency temp | `0.3` | `step2_grid_search_extraction_en.py:106` | |
| Majority threshold | `2` of 3 | `step2_grid_search_extraction_en.py` | |
| Ensemble agree thr | `0.55` | `step2_grid_search_extraction_en.py` | |
| Max ReAct turns | `5` | `step2_grid_search_extraction_en.py` | |
| Context window | `16384` | `step2_grid_search_extraction_en.py` | |
| Retries | `3` | `step2_grid_search_extraction_en.py` | Exponential: 15 s, 30 s, 60 s |
| SOP text limit | `8000` chars | `step2_grid_search_extraction_en.py` | |
| Agreement threshold | `0.75` | `step2_multi_agent_baseline.py` | Duplicate-collapse ratio |
| SBERT model | `all-MiniLM-L6-v2` | `step3_evaluation_rules.py` | |
| Consensus fraction | `0.70` | `step2_multi_run_eval.py` | Rules kept ≥ 70 % of runs |
| Hallucination proxy | `0.30` | `step2_multi_run_eval.py` | Rules in < 30 % of runs |
| IAAS similarity thr | `0.90` | `step2_multi_run_eval.py` | |
| Gap merge | `15` min | `step5_validate.py` | |
| Alarm on-delay | `5` min | `step5_validate.py` | WARNING only |
| Quarantine fraction | `0.50` | `step5_validate.py` | |

---

## 7. Key Outputs & Metrics

| Layer | Primary metric | Output location |
|---|---|---|
| 1 | Chunk count, text size | `layers/texts/` |
| 2a | F1_strict per model × paradigm | `layers/layer_2/step2_results/` |
| 2b | F1_content, precision, recall | `layers/layer_2/step2_results/multi_run/` |
| 3 | IAAS, Fleiss κ, consensus set | `layers/step3_results/` |
| 4a | Node / relationship counts | Neo4j |

---

## 8. Reproducibility Notes

1. **Seed determinism.** Grid-search runs use `SEED=42` by default. LLM calls at `temperature=0.0` are bitwise reproducible within the same model version and context window.

2. **Checkpoint resume.** Both the grid search and the multi-agent pipeline write a checkpoint registry (`layers/state/experiment_registry.json`, `layers/state/agentic_registry.json`). A run interrupted mid-way will resume from the last completed entry. Pass `--force` to disable this and re-run everything.

3. **Layer 3 is not a standalone script.** `step3_evaluation_rules.py` is imported by `step2_multi_run_eval.py`. Running it directly has no effect; run `step2_multi_run_eval.py --skip-runs` to recompute metrics from existing CSVs without re-invoking the LLM.

4. **Neo4j must be reachable.** Step 4a connects to Neo4j using credentials from `.env`. Confirm the instance is running before executing any Layer 4 script.
