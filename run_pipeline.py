"""
run_pipeline.py

Single-button, end-to-end run of the full pipeline -- every runnable
script under layers/, in dependency order:

    Layer 1   test_agent_1b.py                    (parse data/dataset/rules
              PDFs -> texts/; skipped if texts/*.txt already exist)
    Layer 2a  baseline.py                         (deterministic B0 baseline;
              output copied into step2_results/ so Layer 3 scores it)
    Layer 2b  step2_grid_search_extraction_en.py  (model x paradigm grid
              search; HOURS on first run -- runs already completed in its
              registry are skipped on later presses via --no-force)
    Layer 2c  step2_multi_agent_baseline.py       (fresh multi-agent
              extraction, --force)
    Layer 2d  visualize_pipeline.py               (architecture figure)
    Layer 3   step3_evaluation_rules.py           (batch mode: scores every
              CSV in step2_results/)
    Layer 4a  step4_populate.py                   (extracted rules -> Neo4j)
    Layer 4b  step4b_load_abox.py                 (ground-truth ABox -> Neo4j)
    Layer 5   step5_detect.py                     (anomaly-detection report)
    Layer 5b  step5_sensitivity.py                (detector-constant
              sensitivity analysis, read-only)
    Layer 5c  step5_leakage_audit.py              (runtime proof that
              detection never reads ground truth, read-only)

Two things must already be running before you press this button:
  - Ollama Cloud, reachable at OLLAMA_BASE_URL with OLLAMA_API_KEY (checked before anything runs)
  - Neo4j, reachable at NEO4J_URI (checked before anything runs)
This script does not attempt to start either -- both are external services
with their own startup UI/CLI, and failing fast with a clear message is
safer than guessing how to launch them on this machine.

Python packages are also checked up front (requirements.txt): if any
are missing, the script aborts with a pip install command instead of dying
mid-pipeline with an ImportError hours in. It deliberately does not install
anything itself.

Usage
-----
    python run_pipeline.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.request

# (import module name, pip install name) for every third-party package any
# layer script imports. Checked before anything else runs -- including this
# script's own `dotenv` import below -- so a missing package aborts with an
# install command instead of an ImportError traceback mid-pipeline.
_REQUIRED_PACKAGES = [
    ("dotenv", "python-dotenv"),
    ("openai", "openai"),
    ("langgraph", "langgraph"),
    ("docling", "docling"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sentence_transformers", "sentence-transformers"),
    ("neo4j", "neo4j"),
]

_missing = [pip for mod, pip in _REQUIRED_PACKAGES if importlib.util.find_spec(mod) is None]
if _missing:
    print("[ABORT] Missing required Python package(s): " + ", ".join(_missing))
    print("  Install everything the pipeline needs with:")
    print(f"    {sys.executable} -m pip install -r requirements.txt")
    sys.exit(1)

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))

LAYER_1 = os.path.join(_ROOT, "layers", "layer_1")
LAYER_2 = os.path.join(_ROOT, "layers", "layer_2")
LAYER_3 = os.path.join(_ROOT, "layers", "layer_3")
LAYER_4 = os.path.join(_ROOT, "layers", "layer_4")
TEXTS_DIR = os.path.join(LAYER_1, "texts")
STEP2_RESULTS = os.path.join(LAYER_2, "step2_results")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
OLLAMA_API_KEY  = os.environ.get("OLLAMA_API_KEY", "")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _fail(message: str) -> None:
    print(f"\n[ABORT] {message}")
    sys.exit(1)


def _check_ollama() -> None:
    if not OLLAMA_API_KEY:
        _fail("OLLAMA_API_KEY is not set. Add it to the project's .env "
              "(get a key at https://ollama.com/settings/keys).")
    url = OLLAMA_BASE_URL.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
        print(f"  OK  Ollama Cloud reachable at {OLLAMA_BASE_URL}")
    except Exception as exc:
        _fail(
            f"Ollama Cloud is not reachable at {OLLAMA_BASE_URL} ({exc}).\n"
            f"  Check OLLAMA_BASE_URL / OLLAMA_API_KEY in .env and your network."
        )


def _check_neo4j() -> None:
    from neo4j import GraphDatabase

    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "neo4j")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        print(f"  OK  Neo4j reachable at {NEO4J_URI}")
    except Exception as exc:
        _fail(
            f"Neo4j is not reachable at {NEO4J_URI} ({exc}).\n"
            f"  Start your Neo4j instance before running this."
        )


def _run(cmd: list[str], cwd: str) -> None:
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        _fail(f"Step failed (exit {proc.returncode}) after {elapsed:.0f}s: {' '.join(cmd)}")
    print(f"  (done in {elapsed:.0f}s)")


def _latest_rules_csv() -> str:
    candidates = [
        os.path.join(STEP2_RESULTS, f)
        for f in os.listdir(STEP2_RESULTS)
        if f.startswith("ext_multi_agent_langgraph_") and f.endswith(".csv")
    ]
    if not candidates:
        _fail(f"No ext_multi_agent_langgraph_*.csv found in {STEP2_RESULTS} after Layer 2 ran.")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    _banner("PRE-FLIGHT CHECKS")
    print(f"  OK  All {len(_REQUIRED_PACKAGES)} required Python packages present (checked at startup).")
    _check_ollama()
    _check_neo4j()

    _banner("LAYER 1 - Document Parsing")
    existing = (
        [f for f in os.listdir(TEXTS_DIR) if f.endswith(".txt")]
        if os.path.isdir(TEXTS_DIR)
        else []
    )
    if existing:
        print(f"  {len(existing)} .txt file(s) already in layers/layer_1/texts/ -- skipping re-parse.")
    else:
        _run([sys.executable, "test_agent_1b.py"], cwd=LAYER_1)
        parsed = (
            [f for f in os.listdir(TEXTS_DIR) if f.endswith(".txt")]
            if os.path.isdir(TEXTS_DIR)
            else []
        )
        if not parsed:
            _fail(
                "Layer 1 ran but produced no .txt files in layers/layer_1/texts/. "
                "Check that data/dataset/rules/ contains the source SOP PDFs."
            )
        print(f"  Parsed {len(parsed)} SOP document(s) into layers/layer_1/texts/.")

    _banner("LAYER 2a - Deterministic Baseline (B0)")
    _run([sys.executable, "baseline.py"], cwd=LAYER_2)
    baseline_csv = os.path.join(LAYER_2, "baseline_results", "baseline_b0.csv")
    if os.path.isfile(baseline_csv):
        shutil.copyfile(baseline_csv, os.path.join(STEP2_RESULTS, "baseline_b0.csv"))
        print("  Copied baseline_b0.csv into step2_results/ so Layer 3 scores it too.")

    _banner("LAYER 2b - Grid Search (model x paradigm)")
    print("  NOTE: takes hours on a first run; conditions already completed in the")
    print("  grid-search registry are skipped (--no-force), so re-presses resume.")
    _run([sys.executable, "step2_grid_search_extraction_en.py", "--no-force"], cwd=LAYER_2)

    _banner("LAYER 2c - Multi-Agent Extraction (fresh run)")
    _run([sys.executable, "step2_multi_agent_baseline.py", "--force"], cwd=LAYER_2)
    rules_csv = _latest_rules_csv()
    print(f"  Layer 2 output: {rules_csv}")

    _banner("LAYER 2d - Pipeline Architecture Figure")
    _run([sys.executable, "visualize_pipeline.py"], cwd=LAYER_2)

    _banner("LAYER 3 - Evaluation (batch: every CSV in step2_results/)")
    _run([sys.executable, "step3_evaluation_rules.py"], cwd=LAYER_3)

    _banner("LAYER 4a - Knowledge Graph Population")
    _run([sys.executable, "step4_populate.py", "--rules", rules_csv], cwd=LAYER_4)

    _banner("LAYER 4b - ABox Loading + Rule Validation")
    _run([sys.executable, "step4b_load_abox.py"], cwd=LAYER_4)

    _banner("LAYER 5 - Anomaly Detection")
    _run([sys.executable, "step5_detect.py"], cwd=LAYER_4)

    _banner("LAYER 5b - Sensitivity Analysis (read-only)")
    _run([sys.executable, "step5_sensitivity.py"], cwd=LAYER_4)

    _banner("LAYER 5c - GT-Leakage Audit (read-only)")
    _run([sys.executable, "step5_leakage_audit.py"], cwd=LAYER_4)

    _banner("PIPELINE COMPLETE")
    print(f"  Rules CSV : {rules_csv}")
    print(f"  Neo4j     : {NEO4J_URI}")
    print(f"  Evaluation: layers/step3_results/")
    print(f"  Figure    : layers/layer_2/pipeline_full.png")
    print(f"  Reports   : layers/layer_4/detection_results/")


if __name__ == "__main__":
    main()
