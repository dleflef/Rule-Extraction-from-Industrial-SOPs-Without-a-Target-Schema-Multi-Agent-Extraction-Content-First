"""
step2_trulens_eval.py
=====================
TruLens evaluation harness for the multi-agent SOP extraction pipeline.

Evaluates across four dimensions:
  Groundedness     — are final extracted rules supported by the source SOP text?
  Answer Relevance — do extracted rules actually answer the extraction task?
  Context Relevance— is the SOP text relevant to the extraction query?
  Miner Groundedness (per-node) — groundedness of the specialized miner output
                                  (SemanticMiner / ThresholdMiner / AccessMiner)
                                  before merger/validator/normalizer stages.

Results are stored in a local SQLite DB and viewable in the TruLens dashboard.

Install (once — requires Python 3.9; TruLens 2.x requires Python 3.10+):
  pip3 install 'trulens<2.0'

Configure (optional — env vars, .env is auto-loaded):
  TRULENS_FEEDBACK_MODEL   model for feedback LLM (default: gemma3:12b)
  TRULENS_FEEDBACK_TIMEOUT timeout in seconds for feedback LLM calls (default: 120)
  OLLAMA_BASE_URL          OpenAI-compatible endpoint (read from .env)
  OLLAMA_API_KEY           API key for that endpoint   (read from .env)

Usage:
  python step2_trulens_eval.py                          # all SOPs, full + miner eval
  python step2_trulens_eval.py --sop SOP-001.txt        # single SOP
  python step2_trulens_eval.py --reset-db               # wipe DB and re-run
  python step2_trulens_eval.py --dashboard              # open Streamlit dashboard
  python step2_trulens_eval.py --miner-only             # skip full pipeline, miner only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"
))

# ── Pipeline imports ────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from step2_TEST import (  # noqa: E402
    build_pipeline,
    _make_initial_state,
    node_document_type_classifier,
    node_rule_miner,
    llm_call,
    TEXTS_DIR,
    MODEL_NODE_EVALUATOR,
    MODEL_NODE_MINER_SEMANTIC,
    MODEL_NODE_MINER_THRESHOLD,
    MODEL_NODE_MINER_ACCESS,
    _OLLAMA_BASE_URL,
)

# ── TruLens imports ─────────────────────────────────────────────────────────────

try:
    from trulens.core import TruSession, Feedback
    from trulens.core.schema import Select
    # TruLens 1.x path (Python 3.9 compatible; 2.x requires Python 3.10+)
    from trulens.apps.custom import TruCustomApp, instrument
except ImportError as e:
    print(
        f"TruLens not installed or wrong version ({e}).\n"
        "Run:  pip3 install 'trulens<2.0'\n"
        "Note: TruLens 2.x requires Python 3.10+; you are on Python 3.9."
    )
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────────

_DB_PATH = os.path.join(_HERE, "trulens_eval.sqlite")

# Model used to score extracted rules (reuses the pipeline's evaluator model)
_FEEDBACK_MODEL = os.environ.get("TRULENS_FEEDBACK_MODEL", MODEL_NODE_EVALUATOR)

# The extraction task description used as the "query" in relevance feedback
_EXTRACTION_QUERY = (
    "Extract all rules present in this industrial SOP document. Depending on the document "
    "type, rules may include any combination of threshold rules (sensor alarm limits), "
    "operational rules (condition→action pairs), maintenance rules (predictive checks, "
    "cross-station fault chains), or access rules (personnel permissions, occupancy limits, "
    "acknowledgment times). Only rule types that are actually present in the document are "
    "expected — do not penalize for rule types that do not exist in this specific document."
)

# ── Helpers ──────────────────────────────────────────────────────────────────────

def _rules_to_eval_text(rules: list[dict]) -> str:
    """Render rule dicts as natural language sentences for feedback evaluation."""
    if not rules:
        return "No rules were extracted from this document."
    lines = []
    for r in rules:
        rule_id   = r.get("ruleId") or "UNKNOWN"
        cls       = r.get("class") or "UNKNOWN"
        station   = r.get("station") or "N/A"
        condition = r.get("condition") or ""
        action    = r.get("action") or ""
        severity  = r.get("severity") or ""
        lines.append(
            f"[{rule_id}] {cls} at {station} (severity={severity}): "
            f"if {condition}, then {action}."
        )
    return "\n".join(lines)


# ── Instrumented pipeline wrappers ───────────────────────────────────────────────
# Each TruCustomApp must wrap its OWN class instance. If both wraps share the same
# object, the second TruCustomApp's @instrument calls overwrite the first's, so the
# first app's method calls are never captured in TruLens records.

class PipelineApp:
    """TruLens-instrumented wrapper for the full extraction pipeline."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    @instrument
    def extract(self, raw_text: str, source_file: str = "sop") -> str:
        """Full pipeline extraction — input: SOP text, output: rules as text.

        The return value includes both rule_nodes (final_rules) and a summary of
        graph_edges (authorized_for relationships) so that answer-relevance scoring
        can see the full extraction output, not just the non-authorization rule nodes.
        """
        state  = _make_initial_state(raw_text, source_file, "trulens_eval")
        result = self.pipeline.invoke(state)
        rule_text = _rules_to_eval_text(result.get("final_rules", []))
        edges = result.get("graph_edges", [])
        if edges:
            edge_lines = [
                f"  [{e.get('ruleId','?')}] {e.get('condition','?')} → zone={e.get('station','?')}: {e.get('action','?')}"
                for e in edges[:60]
            ]
            rule_text += (
                f"\n\n[AUTHORIZATION EDGES — {len(edges)} role×zone permissions routed to graph]\n"
                + "\n".join(edge_lines)
            )
        return rule_text


class MinerApp:
    """TruLens-instrumented wrapper for the isolated specialized miner nodes.

    Calls node_document_type_classifier then node_rule_miner, which dispatches
    internally to SemanticMiner (A), ThresholdMiner (B), AccessMiner (C), or
    both A+B for mixed documents — mirrors the pipeline routing exactly.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    @instrument
    def mine_rules(self, raw_text: str) -> str:
        """Classify the document then dispatch to the appropriate extractor(s)."""
        state      = _make_initial_state(raw_text, "", "trulens_miner")
        classified = node_document_type_classifier(state)
        state      = {**state, **classified}
        result     = node_rule_miner(state)
        all_rules  = result.get("extracted_rules") or []
        return _rules_to_eval_text(all_rules)


# ── Custom feedback helpers ───────────────────────────────────────────────────────
# TruLens's built-in _with_cot_reasons functions require the LLM to return structured
# JSON (ChainOfThoughtResponse). Local Ollama models output plain text instead, causing
# repeated validation errors. These custom functions use the pipeline's own llm_call
# and ask for a plain 0–1 decimal, which every model handles reliably. Each function
# makes a single holistic call per SOP (not one per rule), avoiding rate-limit storms.

import re as _re

def _score_from_response(raw: str) -> float:
    """Extract a 0–1 float from a model response that may contain reasoning text."""
    raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    # Accept 0.0–1.0 or 0–10 scale (normalise >1 values)
    m = _re.search(r'\b([0-9](?:\.[0-9]+)?)\b', raw)
    if m:
        val = float(m.group(1))
        return min(1.0, max(0.0, val / 10.0 if val > 1.0 else val))
    return 0.5


def _groundedness(source: str, extracted_rules: str) -> float:
    """Are the extracted rules supported by the source SOP text?"""
    prompt = (
        "Rate how well the extracted rules are grounded in the source document.\n\n"
        f"SOURCE DOCUMENT (full text):\n{source[:8000]}\n\n"
        f"EXTRACTED RULES:\n{extracted_rules[:3000]}\n\n"
        "Score from 0.0 to 1.0:\n"
        "  1.0 = every rule is explicitly stated in the source\n"
        "  0.5 = roughly half the rules are supported\n"
        "  0.0 = rules appear fabricated / not in the document\n"
        "Reply with ONLY a single decimal number, e.g. 0.8"
    )
    try:
        raw, _ = llm_call(_FEEDBACK_MODEL, [{"role": "user", "content": prompt}])
        return _score_from_response(raw)
    except Exception:
        return 0.5


def _answer_relevance(extracted_rules: str) -> float:
    """Do the extracted rules adequately answer the extraction task?"""
    prompt = (
        f"TASK: {_EXTRACTION_QUERY}\n\n"
        f"RESPONSE:\n{extracted_rules[:2000]}\n\n"
        "Does the response adequately complete the task? Score from 0.0 to 1.0:\n"
        "  1.0 = fully answers the task with correct rule types and fields\n"
        "  0.5 = partially answers — some rules present but incomplete\n"
        "  0.0 = completely misses the task\n"
        "Reply with ONLY a single decimal number, e.g. 0.9"
    )
    try:
        raw, _ = llm_call(_FEEDBACK_MODEL, [{"role": "user", "content": prompt}])
        return _score_from_response(raw)
    except Exception:
        return 0.5


def _context_relevance(raw_text: str) -> float:
    """Is the SOP document relevant to industrial rule extraction?"""
    prompt = (
        f"TASK: {_EXTRACTION_QUERY}\n\n"
        f"DOCUMENT (excerpt):\n{raw_text[:1500]}\n\n"
        "Is this document relevant to the task? Score from 0.0 to 1.0:\n"
        "  1.0 = document directly contains industrial SOP rules\n"
        "  0.0 = document has no relevant content\n"
        "Reply with ONLY a single decimal number, e.g. 0.95"
    )
    try:
        raw, _ = llm_call(_FEEDBACK_MODEL, [{"role": "user", "content": prompt}])
        return _score_from_response(raw)
    except Exception:
        return 0.5


def _miner_groundedness(source: str, mined_rules: str) -> float:
    """Are the specialized miners' raw rules grounded (before merge/validation)?"""
    prompt = (
        "Rate how well the mined rules (before any merging or validation) are grounded "
        "in the source document.\n\n"
        f"SOURCE DOCUMENT (full text):\n{source[:8000]}\n\n"
        f"MINED RULES:\n{mined_rules[:3000]}\n\n"
        "Score from 0.0 to 1.0 where 1.0 = all rules clearly stated in the source. "
        "Reply with ONLY a single decimal number, e.g. 0.7"
    )
    try:
        raw, _ = llm_call(_FEEDBACK_MODEL, [{"role": "user", "content": prompt}])
        return _score_from_response(raw)
    except Exception:
        return 0.5


# ── Feedback factory ─────────────────────────────────────────────────────────────

def _make_pipeline_feedbacks() -> list[Feedback]:
    f_groundedness = (
        Feedback(_groundedness, name="Groundedness")
        .on(Select.RecordCalls.extract.args.raw_text)
        .on(Select.RecordCalls.extract.rets)
    )
    f_answer_relevance = (
        Feedback(_answer_relevance, name="Answer Relevance")
        .on(Select.RecordCalls.extract.rets)
    )
    f_context_relevance = (
        Feedback(_context_relevance, name="Context Relevance")
        .on(Select.RecordCalls.extract.args.raw_text)
    )
    return [f_groundedness, f_answer_relevance, f_context_relevance]


def _make_miner_feedbacks() -> list[Feedback]:
    f_miner_groundedness = (
        Feedback(_miner_groundedness, name="Miner Groundedness")
        .on(Select.RecordCalls.mine_rules.args.raw_text)
        .on(Select.RecordCalls.mine_rules.rets)
    )
    return [f_miner_groundedness]


# ── Evaluation runner ────────────────────────────────────────────────────────────

def run_evaluation(
    sop_filter: str | None = None,
    reset_db: bool = False,
    miner_only: bool = False,
    app_version: str = "step2_TEST",
) -> None:

    session = TruSession(database_url=f"sqlite:///{_DB_PATH}")
    if reset_db:
        session.reset_database()
        print("TruLens database reset.")

    pipeline_feedbacks = _make_pipeline_feedbacks()
    miner_feedbacks    = _make_miner_feedbacks()

    pipeline     = build_pipeline()
    pipeline_app = PipelineApp(pipeline)
    miner_app    = MinerApp(pipeline)

    # feedback_mode="with_app": run each feedback LLM call synchronously right after
    # the app call completes, before moving to the next SOP. This is slower than the
    # default "deferred" mode but guarantees scores are in the DB before the script exits.
    tru_pipeline = TruCustomApp(
        pipeline_app,
        app_name="SOP Extraction Pipeline",
        app_version=app_version,
        feedbacks=pipeline_feedbacks,
        feedback_mode="with_app",
    )

    tru_miner = TruCustomApp(
        miner_app,
        app_name="SOP Specialized Miners (per-node)",
        app_version=app_version,
        feedbacks=miner_feedbacks,
        feedback_mode="with_app",
    )

    # Discover SOP files
    abs_texts = os.path.abspath(TEXTS_DIR)
    if not os.path.isdir(abs_texts):
        print(f"Texts directory not found: {abs_texts}"); return

    txt_files = sorted(f for f in os.listdir(abs_texts) if f.endswith(".txt"))
    if sop_filter:
        txt_files = [f for f in txt_files if sop_filter in f]
    if not txt_files:
        print(f"No matching .txt files in {abs_texts}"); return

    print(f"\n{'='*60}")
    print(f"  TruLens Evaluation")
    print(f"  Feedback model : {_FEEDBACK_MODEL} @ {_OLLAMA_BASE_URL}")
    print(f"  SOP files      : {txt_files}")
    print(f"  DB             : {_DB_PATH}")
    print(f"  Mode           : {'miner-only' if miner_only else 'full + miner'}")
    print(f"{'='*60}\n")

    for filename in txt_files:
        filepath = os.path.join(abs_texts, filename)
        print(f">> {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if not miner_only:
            with tru_pipeline as recording:
                extract_result = pipeline_app.extract(raw_text, source_file=filename)
            n_rules = len([l for l in extract_result.splitlines() if l.strip()])
            print(f"   full pipeline  → {n_rules} rules recorded")

            # Quick offline score — detect low-quality extraction and trigger reflexion
            quick_g  = _groundedness(raw_text, extract_result)
            quick_ar = _answer_relevance(extract_result)
            print(f"   quick scores   → Groundedness={quick_g:.2f}  Answer Relevance={quick_ar:.2f}")

            if quick_g < 0.8 or quick_ar < 0.8:
                print(f"   [Reflexion] Score below 0.8 — retrying with strict grounding mode …")
                with tru_pipeline as recording:
                    reflexion_result = pipeline_app.extract(
                        raw_text,
                        source_file=filename + "__reflexion",
                    )
                n_ref = len([l for l in reflexion_result.splitlines() if l.strip()])
                ref_g  = _groundedness(raw_text, reflexion_result)
                ref_ar = _answer_relevance(reflexion_result)
                print(f"   reflexion      → {n_ref} rules | G={ref_g:.2f} AR={ref_ar:.2f}")

        with tru_miner as recording:
            miner_result = miner_app.mine_rules(raw_text)
        n_mined = len([l for l in miner_result.splitlines() if l.strip()])
        print(f"   miner-only     → {n_mined} rules recorded\n")

    # Summary — feedback runs in a background thread; wait briefly then query
    print(f"\n{'='*60}")
    print("Evaluation summary (feedback may still be computing…)")
    print(f"{'='*60}")

    import time as _time
    _time.sleep(5)  # give the background feedback thread a moment to finish

    apps_to_check = []
    if not miner_only:
        apps_to_check.append(tru_pipeline)
    apps_to_check.append(tru_miner)

    for tru_app in apps_to_check:
        try:
            # get_records_and_feedback expects a list of app_id strings, not objects
            records, feedback_cols = session.get_records_and_feedback([tru_app.app_id])
            if records.empty:
                print(f"\n  {tru_app.app_name}: no records yet")
                continue
            print(f"\n  {tru_app.app_name}")

            # Aggregate across all records
            for col in feedback_cols:
                if col in records.columns:
                    vals = records[col].dropna()
                    if not vals.empty:
                        print(f"    {col:<30} mean={vals.mean():.3f}  min={vals.min():.3f}  max={vals.max():.3f}")
                    else:
                        print(f"    {col:<30} (still computing)")

            # Per-SOP breakdown — shows which document drives any low scores
            if "input" in records.columns and feedback_cols:
                print(f"\n    Per-SOP breakdown:")
                for idx, row in records.iterrows():
                    # TruLens stores the first arg in 'input'; we use app_name as label if unavailable
                    label = str(row.get("input", ""))[:40] or f"record_{idx}"
                    scores = "  ".join(
                        f"{col}={row[col]:.2f}" for col in feedback_cols
                        if col in records.columns and row.get(col) is not None
                        and str(row.get(col)) not in ("", "nan")
                    )
                    flag = "  ← low" if any(
                        str(row.get(col, "")) not in ("", "nan")
                        and float(row.get(col, 1.0)) < 0.8
                        for col in feedback_cols if col in records.columns
                    ) else ""
                    print(f"      {label:<42} {scores}{flag}")
        except Exception as exc:
            print(f"  (Could not retrieve records: {exc})")

    print(f"\nDashboard: python step2_trulens_eval.py --dashboard")
    print(f"DB path  : {_DB_PATH}")


# ── Dashboard launcher ────────────────────────────────────────────────────────────

def launch_dashboard() -> None:
    import importlib.util
    import subprocess

    session = TruSession(database_url=f"sqlite:///{_DB_PATH}")
    print(f"Launching TruLens dashboard (DB: {_DB_PATH}) …")

    try:
        # Preferred path: TruLens built-in launcher (needs 'streamlit' on PATH)
        session.run_dashboard()
    except FileNotFoundError:
        # 'streamlit' binary not in PATH — launch via python3 -m streamlit instead
        spec = importlib.util.find_spec("trulens.dashboard")
        if spec is None or spec.origin is None:
            print("trulens.dashboard package not found.")
            return
        dashboard_dir = os.path.dirname(spec.origin)
        # TruLens 1.x entry point is Leaderboard.py in the dashboard package dir
        app_path = os.path.join(dashboard_dir, "Leaderboard.py")
        if not os.path.exists(app_path):
            # Fallback: look one level up (some versions nest differently)
            for candidate in ["Leaderboard.py", "app.py", "streamlit_app.py"]:
                p = os.path.join(dashboard_dir, candidate)
                if os.path.exists(p):
                    app_path = p
                    break
            else:
                print(f"Could not find dashboard app in {dashboard_dir}")
                return
        db_url = f"sqlite:///{_DB_PATH}"
        print(f"  → using python3 -m streamlit run {app_path}")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--", f"--database-url={db_url}",
        ])


# ── CLI ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruLens evaluation for SOP extraction pipeline")
    parser.add_argument("--sop",        default=None,  help="Filter to SOPs whose filename contains this string")
    parser.add_argument("--reset-db",   action="store_true", help="Drop and recreate the TruLens database")
    parser.add_argument("--miner-only", action="store_true", help="Run miner-level eval only (skip full pipeline)")
    parser.add_argument("--dashboard",  action="store_true", help="Launch TruLens Streamlit dashboard")
    parser.add_argument("--version",    default="step2_TEST", help="App version label in TruLens (default: step2_TEST)")
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
    else:
        run_evaluation(
            sop_filter=args.sop,
            reset_db=args.reset_db,
            miner_only=args.miner_only,
            app_version=args.version,
        )
